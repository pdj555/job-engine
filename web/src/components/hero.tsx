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
  const done = resultCount > 0;

  return (
    <section className="hero-grid">
      <div className="min-w-0">
        <h1>
          <span className="title-text">Job Engine</span>
        </h1>

        <Panel label="Search Progress" className="mt-5">
          <div className="meta-row">
            <span>
              {done ? `${resultCount} ranked` : "0 results"}
              {done ? ` / ${resultCount}` : " / —"}
            </span>
            <span>{done ? "100%" : "—"}</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: done ? "100%" : "0%" }} />
          </div>
          <div className="meta-row mt-3 mb-0">
            <span>{done ? "run complete" : "idle"}</span>
            <span>pipeline {pipelinePct}%</span>
          </div>
        </Panel>

        <p className="about">
          About Job Engine: search contracts, grants, and roles across sources. Each
          listing is ranked by effective hourly rate — annual compensation divided by
          expected hours — so opportunities compare on equal footing. Track applications
          in your pipeline from search to offer.
        </p>
      </div>

      <Panel label="Live Global Status">
        <WorldMap live={live} />
      </Panel>
    </section>
  );
}
