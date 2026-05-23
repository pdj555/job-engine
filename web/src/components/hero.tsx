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
      <div className="min-w-0 flex flex-col gap-5">
        <h1>
          <span className="title-text">Job Engine</span>
        </h1>

        <p className="about">
          Search contracts, grants, and roles across sources. Every listing is ranked by
          effective hourly rate — annual compensation divided by expected hours — so
          opportunities compare on equal footing. Track applications from search to offer.
        </p>

        <Panel label="Search Progress" className="mt-auto">
          <div className="meta-row">
            <span>{done ? `${resultCount} ranked / ${resultCount}` : "0 results / —"}</span>
            <span>{done ? "100%" : "—"}</span>
          </div>
          <div className="bar-track mt-2">
            <div className="bar-fill" style={{ width: done ? "100%" : "0%" }} />
          </div>
          <div className="meta-row mt-2">
            <span>{done ? "run complete" : "idle"}</span>
            <span>pipeline {pipelinePct}%</span>
          </div>
        </Panel>
      </div>

      <Panel label="Live Global Status" className="flex flex-col">
        <WorldMap live={live} />
      </Panel>
    </section>
  );
}
