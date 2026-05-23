"use client";

import { NodeMap } from "./node-map";
import { Panel } from "./panel";

export function Hero({
  pipelinePct,
  live,
}: {
  pipelinePct: number;
  live: boolean;
}) {
  return (
    <section className="grid lg:grid-cols-[1fr_13rem] gap-6 lg:gap-8 items-start">
      <div className="min-w-0">
        <h1 className="title">
          <span className="title-text">Job Engine</span>
        </h1>

        <Panel label="Search Progress" className="mt-6 lg:mt-8">
          <div className="flex justify-between meta mb-3 gap-4">
            <span>Query → rank → pipeline</span>
            <span className="shrink-0">{pipelinePct}% pipeline</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${pipelinePct}%` }} />
          </div>
        </Panel>

        <p className="prose mt-6 max-w-xl">
          Search contracts, grants, and roles. Results ranked by effective hourly
          rate — pay divided by expected hours, not title prestige.
        </p>
      </div>

      <Panel label="Live Global Status" className="min-w-0">
        <NodeMap live={live} />
      </Panel>
    </section>
  );
}
