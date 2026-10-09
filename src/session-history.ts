import { access, readdir } from "node:fs/promises";
import path from "node:path";
import { claudeConfigDir } from "./paths.js";
import {
  getSessionMessages,
  importSessionToStore,
  type SessionMessage,
  type SessionStoreEntry,
} from "@anthropic-ai/claude-agent-sdk";

/** Read-only compatibility for SDK versions that ignore explicit last-prompt
 * anchors. The CLI owns all transcript writes, including an empty-history anchor. */
export async function readSessionHistory(
  sessionId: string,
  includeSystemMessages = false,
): Promise<SessionMessage[]> {
  const messages = includeSystemMessages
    ? await getSessionMessages(sessionId, { includeSystemMessages: true })
    : await getSessionMessages(sessionId);
  if (messages.length === 0) return messages;
  // Remote/fileless SDK sessions have no local JSONL anchor to interpret.
  // A failed read of an existing transcript must still propagate.
  if (!/^[A-Za-z0-9_-]+$/.test(sessionId)) throw new Error("Invalid session id for local history");
  const projects = path.join(claudeConfigDir(), "projects");
  const directories = await readdir(projects).catch((error: { code?: string }) => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  let local = false;
  for (const directory of directories) {
    try {
      await access(path.join(projects, directory, `${sessionId}.jsonl`));
      local = true;
      break;
    } catch (error) {
      if ((error as { code?: string }).code !== "ENOENT") throw error;
    }
  }
  if (!local) return messages;
  const rows: SessionStoreEntry[] = [];
  await importSessionToStore(
    sessionId,
    {
      append: async (_, entries) => {
        rows.push(...entries);
      },
      load: async () => null,
    },
    { includeSubagents: false },
  );
  return applyRewindAnchor(messages, rows);
}

export function applyRewindAnchor(
  messages: SessionMessage[],
  rows: SessionStoreEntry[],
): SessionMessage[] {
  let anchor: string | null | undefined;
  let anchorIndex = -1;
  const byId = new Map<string, { row: SessionStoreEntry; index: number }>();
  // The CLI may resume from the prompt snapshot after the retained assistant.
  // Only snapshots written before the latest rewind may connect to the current leaf.
  function snapshotBridge(parent: unknown, leaf: string | null): boolean {
    if (leaf === null) return false;
    const seen = new Set<string>();
    while (parent !== leaf) {
      if (typeof parent !== "string" || seen.has(parent)) return false;
      seen.add(parent);
      const entry = byId.get(parent);
      if (!entry || entry.index >= anchorIndex) return false;
      const { row } = entry;
      const attachment = row.attachment;
      if (
        row.type !== "attachment" ||
        typeof attachment !== "object" ||
        attachment === null ||
        !("type" in attachment) ||
        attachment.type !== "prompt_snapshot"
      ) {
        return false;
      }
      parent = row.parentUuid;
    }
    return true;
  }
  for (const [index, row] of rows.entries()) {
    if (row.isSidechain || row.parent_tool_use_id != null || row.parent_agent_id != null) continue;
    if (
      row.uuid &&
      "parentUuid" in row &&
      ["user", "assistant", "system", "attachment"].includes(row.type)
    ) {
      const known = byId.has(row.uuid);
      byId.set(row.uuid, { row, index });
      if (anchor !== undefined && !known) {
        if (row.parentUuid === anchor || snapshotBridge(row.parentUuid, anchor)) anchor = row.uuid;
        else if (row.type === "user" || row.type === "assistant") {
          throw new Error(
            "Conversation changed outside the retained rewind chain; reload required",
          );
        }
      }
    } else if (
      row.type === "last-prompt" &&
      row.explicit === true &&
      (row.leafUuid === null || typeof row.leafUuid === "string")
    ) {
      anchor = row.leafUuid;
      anchorIndex = index;
    }
  }
  if (anchor === undefined) return messages;
  if (anchor === null) return [];
  const retained = new Set<string>();
  let cursor: string | null = anchor;
  while (cursor !== null) {
    if (retained.has(cursor)) throw new Error("Cycle in the retained conversation chain");
    const row: SessionStoreEntry | undefined = byId.get(cursor)?.row;
    if (!row) throw new Error("Rewind anchor chain is unavailable; refusing stale history replay");
    retained.add(cursor);
    cursor = typeof row.parentUuid === "string" ? row.parentUuid : null;
  }
  return messages.filter((message) => retained.has(message.uuid));
}
