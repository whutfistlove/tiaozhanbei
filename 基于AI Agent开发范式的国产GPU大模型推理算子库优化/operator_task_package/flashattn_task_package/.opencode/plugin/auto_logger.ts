/**
 * OpenCode event logger.
 *
 * This hook records tool-level evidence only.  Formal A/B optimization results
 * are written by agent_system.optimization_log into runs/<run_id>/logs/.
 */
import type { Plugin } from "@opencode-ai/plugin"

export default (async ({ project }) => {
  const root = project?.cwd || "."
  const eventsPath = `${root}/results/opencode_events.jsonl`

  return {
    "tool.execute.after": async (input, output) => {
      const tool = input?.info?.tool
      if (tool !== "bash" && tool !== "edit") return

      const entry: Record<string, unknown> = {
        timestamp: new Date().toISOString(),
        run_id: process.env.AGENT_RUN_ID || "opencode",
        tool,
        agent: input?.info?.agent_id || "unknown",
      }

      if (tool === "bash") {
        const cmd = (input as any)?.args?.command || ""
        const interesting =
          cmd.includes("run_closed_loop.py") ||
          cmd.includes("mxcc") ||
          cmd.includes("pytest") ||
          cmd.includes("benchmark")
        if (!interesting) return
        entry.command = cmd.slice(0, 500)
        entry.exit_code = (output as any)?.code
      }

      if (tool === "edit") {
        const fp = (input as any)?.args?.filePath || ""
        const interesting =
          fp.endsWith(".cu") ||
          fp.includes("/agent_system/") ||
          fp.includes("\\agent_system\\") ||
          fp.includes("/.opencode/") ||
          fp.includes("\\.opencode\\")
        if (!interesting) return
        entry.file = fp
      }

      appendJsonl(eventsPath, entry)
    },
  }
}) satisfies Plugin

function appendJsonl(path: string, entry: Record<string, unknown>) {
  try {
    const fs = require("fs")
    const dir = require("path").dirname(path)
    fs.mkdirSync(dir, { recursive: true })
    fs.appendFileSync(path, JSON.stringify(entry) + "\n")
  } catch {
    // Logging must never block the optimization loop.
  }
}
