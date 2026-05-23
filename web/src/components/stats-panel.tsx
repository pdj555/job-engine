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
    <div className="space-y-4 lg:sticky lg:top-14">
      <Panel label="Live Status">
        <div className="flex items-center gap-2 mb-4 pb-3 border-b border-[var(--accent-soft)]">
          <span
            className={`w-2 h-2 rounded-full bg-[var(--accent)] ${
              apiOnline ? "pulse" : "opacity-25"
            }`}
          />
          <span className="text-[10px] uppercase tracking-widest">
            API {apiOnline === null ? "..." : apiOnline ? "online" : "offline"}
          </span>
        </div>
        <dl className="grid grid-cols-2 gap-3">
          <div className="border border-[var(--accent-soft)] p-2">
            <dt className="stat-label">Avg $/hr</dt>
            <dd className="stat-value text-lg">{avgRate != null ? `$${avgRate}` : "—"}</dd>
          </div>
          <div className="border border-[var(--accent-soft)] p-2">
            <dt className="stat-label">Remote</dt>
            <dd className="stat-value text-lg">{remotePct != null ? `${remotePct}%` : "—"}</dd>
          </div>
          <div className="border border-[var(--accent-soft)] p-2">
            <dt className="stat-label">Results</dt>
            <dd className="stat-value text-lg">{results.length}</dd>
          </div>
          <div className="border border-[var(--accent-soft)] p-2">
            <dt className="stat-label">Pipeline</dt>
            <dd className="stat-value text-lg">{completionPct}%</dd>
          </div>
        </dl>
      </Panel>

      {rates.length > 0 && (
        <Panel label="Compensation Evaluations">
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
