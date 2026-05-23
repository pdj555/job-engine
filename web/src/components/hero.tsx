"use client";

import { NodeMap } from "./node-map";
import { Panel } from "./panel";

export function Hero({
  resultCount,
  pipelinePct,
  apiOnline,
}: {
  resultCount: number;
  pipelinePct: number;
  apiOnline: boolean | null;
}) {
  const complete = resultCount > 0;
  const status = apiOnline === null ? "..." : apiOnline ? "online" : "offline";

  return (
    <section className="grid lg:grid-cols-[1fr_260px] gap-8 items-start">
      <div>
        <h1 className="title">Job Engine</h1>

        <Panel label="Search Progress" className="mt-6">
          <div className="flex justify-between text-[10px] uppercase tracking-widest mb-2">
            <span className="tabular-nums">
              {resultCount > 0 ? `${resultCount} ranked` : "0 results"}
            </span>
            <span className="text-[var(--accent-muted)]">
              {complete ? "run complete" : "idle"}
            </span>
          </div>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${Math.min(resultCount * 5, 100)}%` }}
            />
          </div>
          <p className="mt-3 text-[10px] uppercase tracking-widest text-[var(--accent-muted)]">
            API <span className="text-[var(--fg)]">{status}</span>
            <span className="mx-2 opacity-30">·</span>
            Pipeline <span className="text-[var(--fg)] tabular-nums">{pipelinePct}%</span>
          </p>
        </Panel>

        <p className="mt-6 text-[11px] leading-[1.7] text-[var(--accent-muted)] max-w-xl">
          <span className="text-[var(--fg)] uppercase tracking-widest text-[10px]">
            About Job Engine:{" "}
          </span>
          Find roles, contracts, grants, and equity opportunities ranked by compensation
          per hour. Non-remote roles take a 30% penalty. Track applications in the
          pipeline below.
        </p>
      </div>

      <Panel label="Live Global Status">
        <NodeMap live={complete || apiOnline === true} />
      </Panel>
    </section>
  );
}
