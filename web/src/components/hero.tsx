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
  const progress = resultCount > 0 ? Math.min(resultCount * 5, 100) : 0;
  const live = resultCount > 0 || apiOnline === true;

  return (
    <section className="grid lg:grid-cols-[1fr_240px] gap-8 lg:gap-10 items-start">
      <div>
        <p className="subtitle mb-3">Opportunity search · ranked by $/hr</p>
        <h1 className="title">Job Engine</h1>

        <Panel label="Search Progress" className="mt-8">
          <div className="flex justify-between items-baseline text-[9px] uppercase tracking-[0.12em] mb-3">
            <span className="tabular-nums text-[var(--fg)]">
              {resultCount > 0 ? `${resultCount} ranked` : "Awaiting query"}
            </span>
            <span className="text-[var(--accent-muted)]">
              {resultCount > 0 ? "complete" : "idle"}
            </span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="mt-4 text-[9px] uppercase tracking-[0.12em] text-[var(--accent-muted)]">
            Pipeline completion{" "}
            <span className="text-[var(--fg)] tabular-nums">{pipelinePct}%</span>
          </p>
        </Panel>

        <p className="mt-8 text-[11px] leading-[1.75] text-[var(--accent-muted)] max-w-lg">
          Find roles, contracts, grants, and equity opportunities ranked by
          compensation per hour. Non-remote roles take a 30% penalty. Track
          applications in the pipeline below.
        </p>
      </div>

      <Panel label="Network Status">
        <NodeMap live={live} />
      </Panel>
    </section>
  );
}
