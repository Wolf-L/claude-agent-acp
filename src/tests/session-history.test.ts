import type { SessionMessage, SessionStoreEntry } from "@anthropic-ai/claude-agent-sdk";
import { describe, expect, it } from "vitest";
import { applyRewindAnchor } from "../session-history.js";
const rows: SessionStoreEntry[] = [
  { type: "user", uuid: "u1", parentUuid: null },
  { type: "assistant", uuid: "a1", parentUuid: "u1" },
  { type: "attachment", uuid: "hidden", parentUuid: "a1" },
  { type: "user", uuid: "u2", parentUuid: "hidden" },
  { type: "assistant", uuid: "a2", parentUuid: "u2" },
];
const messages = rows.filter((r) => r.type !== "attachment") as unknown as SessionMessage[];
const anchor = (leafUuid: string | null) => ({
  type: "last-prompt",
  explicit: true,
  rewound: true,
  leafUuid,
});
const snapshot = (uuid: string, parentUuid: string | null): SessionStoreEntry => ({
  type: "attachment",
  uuid,
  parentUuid,
  attachment: { type: "prompt_snapshot" },
});
const edited: SessionStoreEntry = { type: "user", uuid: "edited", parentUuid: "hidden" };
const reply: SessionStoreEntry = { type: "assistant", uuid: "reply", parentUuid: "edited" };
const snapshotRows = [rows[0], rows[1], snapshot("hidden", "a1"), rows[3], rows[4]];
const continuedMessages = [...messages, edited, reply] as SessionMessage[];
describe("native rewind anchor replay", () => {
  it("keeps legacy SDK history unchanged without an explicit anchor", () => {
    expect(applyRewindAnchor(messages, snapshotRows)).toBe(messages);
  });
  it("continues through the existing CLI prompt snapshot after the retained assistant", () => {
    expect(
      applyRewindAnchor(continuedMessages, [...snapshotRows, anchor("a1"), edited, reply]).map(
        (m) => m.uuid,
      ),
    ).toEqual(["u1", "a1", "edited", "reply"]);
  });
  it("bridges multiple pre-anchor snapshots and subsequent hidden continuation records", () => {
    const continuation = [
      { ...edited, parentUuid: "snapshot-2" },
      { type: "attachment", uuid: "reminder", parentUuid: "edited" },
      { ...reply, parentUuid: "reminder" },
    ];
    expect(
      applyRewindAnchor(continuedMessages, [
        ...snapshotRows,
        snapshot("snapshot-2", "hidden"),
        anchor("a1"),
        ...continuation,
      ]).map((m) => m.uuid),
    ).toEqual(["u1", "a1", "edited", "reply"]);
  });
  it("supports a repeated rewind and continuation through its pre-existing snapshot", () => {
    const again = { ...edited, uuid: "edited-again" };
    const againReply = { ...reply, uuid: "reply-again", parentUuid: "edited-again" };
    expect(
      applyRewindAnchor([...continuedMessages, again, againReply] as SessionMessage[], [
        ...snapshotRows,
        anchor("a1"),
        edited,
        reply,
        anchor("a1"),
        again,
        againReply,
      ]).map((m) => m.uuid),
    ).toEqual(["u1", "a1", "edited-again", "reply-again"]);
  });
  it("allows a snapshot written before a later rewind to bridge that later anchor", () => {
    const again = { ...edited, uuid: "edited-again", parentUuid: "new-snapshot" };
    const discarded = { ...edited, uuid: "discarded", parentUuid: "new-snapshot" };
    expect(
      applyRewindAnchor([...continuedMessages, discarded, again] as SessionMessage[], [
        ...snapshotRows,
        anchor("a1"),
        edited,
        reply,
        snapshot("new-snapshot", "reply"),
        discarded,
        anchor("reply"),
        again,
      ]).map((m) => m.uuid),
    ).toEqual(["u1", "a1", "edited", "reply", "edited-again"]);
  });
  it.each(["u2", "a2", "absent", "hidden", null])(
    "rejects a snapshot bridge through discarded, missing, cyclic or root parent %s",
    (parent) => {
      expect(() =>
        applyRewindAnchor(continuedMessages, [
          ...rows,
          snapshot("hidden", parent),
          anchor("a1"),
          edited,
        ]),
      ).toThrow("outside the retained rewind chain");
    },
  );
  it.each([
    { type: "attachment" },
    { type: "attachment", attachment: { type: "environment" } },
    { type: "system", attachment: { type: "prompt_snapshot" } },
    { type: "attachment", attachment: "prompt_snapshot" },
    { type: "attachment", attachment: null },
  ])("does not bridge arbitrary metadata: %j", (metadata) => {
    expect(() =>
      applyRewindAnchor(continuedMessages, [
        ...rows,
        { uuid: "hidden", parentUuid: "a1", ...metadata },
        anchor("a1"),
        edited,
      ]),
    ).toThrow("outside the retained rewind chain");
  });
  it("rejects cycles involving multiple snapshots", () => {
    expect(() =>
      applyRewindAnchor(continuedMessages, [
        ...rows,
        snapshot("hidden", "other-snapshot"),
        snapshot("other-snapshot", "hidden"),
        anchor("a1"),
        edited,
      ]),
    ).toThrow("outside the retained rewind chain");
  });
  it.each(["user", "assistant"])("rejects a competing %s after a valid bridge", (type) => {
    expect(() =>
      applyRewindAnchor(continuedMessages, [
        ...snapshotRows,
        anchor("a1"),
        edited,
        reply,
        { type, uuid: "competing", parentUuid: "hidden" },
        anchor("a1"),
      ]),
    ).toThrow("outside the retained rewind chain");
  });
  it("does not bridge a new competing snapshot back to the original rewind anchor", () => {
    expect(() =>
      applyRewindAnchor(continuedMessages, [
        ...snapshotRows,
        anchor("a1"),
        edited,
        reply,
        snapshot("competing-snapshot", "a1"),
        { ...edited, uuid: "competing", parentUuid: "competing-snapshot" },
      ]),
    ).toThrow("outside the retained rewind chain");
  });
  it("does not trust a snapshot rewritten after the anchor", () => {
    expect(() =>
      applyRewindAnchor(continuedMessages, [
        ...snapshotRows,
        anchor("a1"),
        snapshot("hidden", "a1"),
        edited,
      ]),
    ).toThrow("outside the retained rewind chain");
  });
  it.each([{ isSidechain: true }, { parent_tool_use_id: "tool" }, { parent_agent_id: "agent" }])(
    "does not bridge a sidechain snapshot: %j",
    (sidechain) => {
      expect(() =>
        applyRewindAnchor(continuedMessages, [
          rows[0],
          rows[1],
          { ...snapshot("hidden", "a1"), ...sidechain },
          anchor("a1"),
          edited,
        ]),
      ).toThrow("outside the retained rewind chain");
    },
  );
  it("does not bridge an empty first-message rewind even through a root snapshot", () => {
    expect(() =>
      applyRewindAnchor(continuedMessages, [
        ...snapshotRows,
        snapshot("root-snapshot", null),
        anchor(null),
        { ...edited, parentUuid: "root-snapshot" },
      ]),
    ).toThrow("outside the retained rewind chain");
  });
  it("continues a first-message rewind only from a new root", () => {
    expect(
      applyRewindAnchor(continuedMessages, [
        ...snapshotRows,
        anchor(null),
        { ...edited, parentUuid: null },
        reply,
      ]).map((m) => m.uuid),
    ).toEqual(["edited", "reply"]);
  });
  it("returns empty for first-message anchor even when SDK returns stale messages", () => {
    expect(applyRewindAnchor(messages, [...rows, anchor(null)])).toEqual([]);
  });
  it("walks parents through hidden chain entries instead of a visible index", () => {
    expect(applyRewindAnchor(messages, [...rows, anchor("hidden")]).map((m) => m.uuid)).toEqual([
      "u1",
      "a1",
    ]);
  });
  it("preserves explicit anchors across non-explicit metadata and sidechains", () => {
    expect(
      applyRewindAnchor(messages, [
        ...rows,
        anchor("a1"),
        { type: "last-prompt", leafUuid: "a1" },
        { type: "user", uuid: "child", parentUuid: "a2", isSidechain: true },
      ]).map((m) => m.uuid),
    ).toEqual(["u1", "a1"]);
  });
  it("uses the SDK active chain after a new prompt follows the anchor", () => {
    const current = [
      messages[0],
      messages[1],
      { type: "user", uuid: "edited", parentUuid: "a1" },
    ] as SessionMessage[];
    expect(
      applyRewindAnchor(current, [
        ...rows,
        anchor("a1"),
        { type: "user", uuid: "edited", parentUuid: "a1" },
      ]),
    ).toEqual(current);
  });
  it("does not reactivate stale log entries or duplicate pre-rewind rows", () => {
    expect(
      applyRewindAnchor(messages, [
        ...rows,
        anchor("a1"),
        rows[4],
        { type: "progress", uuid: "late", parentUuid: "a2" },
        { type: "system", uuid: "late-log", parentUuid: "a2" },
      ]).map((m) => m.uuid),
    ).toEqual(["u1", "a1"]);
    expect(() =>
      applyRewindAnchor(messages, [
        ...rows,
        anchor("a1"),
        { type: "assistant", uuid: "late-answer", parentUuid: "a2" },
      ]),
    ).toThrow("outside");
  });
  it("rejects missing and cyclic chains instead of replaying the deleted suffix", () => {
    expect(() => applyRewindAnchor(messages, [...rows, anchor("absent")])).toThrow("unavailable");
    expect(() =>
      applyRewindAnchor(messages, [
        { type: "user", uuid: "loop", parentUuid: "loop" },
        anchor("loop"),
      ]),
    ).toThrow("Cycle");
  });
});
