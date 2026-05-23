"use client";

import type { Opportunity } from "@/lib/types";
import { Panel } from "./panel";
import { LineChart } from "./line-chart";

export function StatsPanel({
  results,
  apiOnline,
  doneCount,
  activeCount,
  completionPct,
}: {
  results: Opportunity[];
  apiOnline: boolean | null;
  doneCount: number;
  activeCount: number;
  completionPct: number;
}) {
  const rates = results
    .map((r) => r.dollars_per_hour)
    .filter((r): r is number => r != null)
    .slice(0, 8);

  const avgRate =
    rates.length > 0 ? Math.round(rates.reduce((a, b) => a + b, 0) / rates.length) : null;

  const remotePct =
    results.length > 0
      ? Math.round((results.filter((r) => r.remote).length / results.length) * 100)
      : null;

  return (
    <div className="space-y-5 lg:sticky lg:top-[4.25rem]">
      <Panel label="Metrics">
        <dl className="grid grid-cols-2 gap-2.5">
          <div className="stat-tile">
            <dt className="stat-label">Avg $/hr</dt>
            <dd className="stat-value text-lg mt-1">
              {avgRate != null ? `$${avgRate}` : "—"}
            </dd>
          </div>
          <div className="stat-tile">
            <dt className="stat-label">Remote</dt>
            <dd className="stat-value text-lg mt-1">
              {remotePct != null ? `${remotePct}%` : "—"}
            </dd>
          </div>
          <div className="stat-tile">
            <dt className="stat-label">Results</dt>
            <dd className="stat-value text-lg mt-1">{results.length}</dd>
          </div>
          <div className="stat-tile">
            <dt className="stat-label">Pipeline</dt>
            <dd className="stat-value text-lg mt-1">{completionPct}%</dd>
          </div>
        </dl>
        <p className="mt-4 pt-3 border-t border-[var(--border)] text-[8px] uppercase tracking-[0.12em] text-[var(--accent-muted)]">
          API {apiOnline === null ? "checking" : apiOnline ? "online" : "offline"}
        </p>
      </Panel>

      {rates.length > 0 && (
        <Panel label="Compensation">
          <LineChart values={rates} title="Top by $/hr" format={(v) => `$${Math.round(v)}`} />
        </Panel>
      )}

      {doneCount + activeCount > 0 && (
        <Panel label="Pipeline Velocity">
          <LineChart
            values={[doneCount, activeCount, doneCount + activeCount]}
            title="Done · Active · Total"
          />
        </Panel>
      )}
    </div>
  );
}
