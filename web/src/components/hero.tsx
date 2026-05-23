"use client";

import { Panel } from "./panel";
import { WorldMap } from "./world-map";

export function Hero({
  resultCount,
  pipelinePct,
  live,
}: {
  resultCount: number;
  pipelinePct: number;
  live: boolean;
}) {
  const progress = resultCount > 0 ? 100 : 0;
  const status = resultCount > 0 ? "run complete" : "idle";

  return (
    <section className="hero-grid">
      <div className="min-w-0">
        <h1>
          <span className="title-text">Job Engine</span>
        </h1>

        <Panel label="Search Progress" className="mt-5">
          <div className="meta-row">
            <span>
              {resultCount > 0 ? `${resultCount} ranked` : "0 results"}
            </span>
            <span>{status}</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="meta-row mt-3 mb-0">
            <span>pipeline {pipelinePct}%</span>
            <span>{resultCount > 0 ? "100%" : "0%"}</span>
          </div>
        </Panel>

        <p className="about">
          About Job Engine: search contracts, grants, and roles across sources.
          Each listing is ranked by effective hourly rate — annual pay divided by
          expected hours — so compensation compares on equal footing. Add roles to
          your pipeline and track applications in one place.
        </p>
      </div>

      <Panel label="Live Global Status">
        <WorldMap live={live} />
      </Panel>
    </section>
  );
}
