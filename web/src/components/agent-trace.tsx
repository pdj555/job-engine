"use client";

import { Panel } from "./panel";

// The agent's autonomous plan, made visible: each line is a search it chose to run.
export function AgentTrace({ searches }: { searches: string[] }) {
  if (searches.length === 0) return null;

  return (
    <Panel label="Agent Trace">
      <div className="meta-row mb-3">
        <span>autonomous</span>
        <span>
          {searches.length} {searches.length === 1 ? "search" : "searches"}
        </span>
      </div>
      <ol className="agent-trace">
        {searches.map((q, i) => (
          <li key={`${i}-${q}`} style={{ animationDelay: `${i * 70}ms` }}>
            <span className="prompt" aria-hidden>
              &gt;
            </span>
            <span className="agent-trace-q">{q}</span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
