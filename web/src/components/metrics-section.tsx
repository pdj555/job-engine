"use client";

import type { Opportunity } from "@/lib/types";
import { formatRate } from "@/lib/format";
import { LineChart } from "./line-chart";

function series(values: number[], points = 16): number[] {
  if (values.length === 0) return Array(points).fill(0);
  const out: number[] = [];
  const size = Math.max(1, values.length / points);
  for (let i = 0; i < points; i++) {
    const start = Math.floor(i * size);
    const slice = values.slice(start, Math.floor((i + 1) * size));
    out.push(slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : 0);
  }
  return out;
}

function MetricCell({ title, value, data }: { title: string; value: string; data: number[] }) {
  return (
    <div className="metric-cell">
      <div className="metric-title">
        {title}: {value}
      </div>
      <LineChart data={data} />
    </div>
  );
}

export function MetricsSection({ results }: { results: Opportunity[] }) {
  if (results.length === 0) return null;

  const rates = results.map((r) => r.dollars_per_hour ?? 0).filter((r) => r > 0);
  const avg = rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : 0;
  const remotePct = Math.round(
    (results.filter((r) => r.remote).length / results.length) * 100
  );
  const top = rates.length ? Math.max(...rates) : 0;
  const count = results.length;

  return (
    <section className="section-head">
      <span className="section-head-label">Compensation Evaluations</span>
      <div className="metric-grid">
        <MetricCell title="Avg $/hr" value={formatRate(avg)} data={series(rates)} />
        <MetricCell title="Remote" value={`${remotePct}%`} data={series(results.map((r) => (r.remote ? 100 : 0)))} />
        <MetricCell title="Top $/hr" value={formatRate(top)} data={series([...rates].sort((a, b) => a - b))} />
        <MetricCell title="Results" value={String(count)} data={series(rates.map((_, i) => i + 1))} />
      </div>
    </section>
  );
}
