"use client";

import type { Opportunity } from "@/lib/types";
import { formatRate } from "@/lib/format";
import { LineChart } from "./line-chart";

function series(values: number[], points = 16): number[] {
  if (values.length === 0) return Array.from({ length: points }, () => 0);
  const out: number[] = [];
  const size = Math.max(1, values.length / points);
  for (let i = 0; i < points; i++) {
    const start = Math.floor(i * size);
    const slice = values.slice(start, Math.floor((i + 1) * size));
    out.push(slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : 0);
  }
  return out;
}

function MetricCell({
  title,
  value,
  data,
  active,
}: {
  title: string;
  value: string;
  data: number[];
  active: boolean;
}) {
  return (
    <div className="metric-cell">
      <div className="metric-title">
        {title} <b className={active ? "on" : ""}>{value}</b>
      </div>
      <LineChart data={data} active={active} />
    </div>
  );
}

export function MetricsSection({ results }: { results: Opportunity[] }) {
  const idle = results.length === 0;
  const rates = results
    .map((r) => r.refined_rate ?? r.dollars_per_hour ?? 0)
    .filter((r) => r > 0);
  const imputed = results.some((r) => r.rate_imputed);
  const avg = rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : 0;
  const remotePct = results.length
    ? Math.round((results.filter((r) => r.remote).length / results.length) * 100)
    : 0;
  const top = rates.length ? Math.max(...rates) : 0;

  return (
    <section className="section-head">
      <span className="section-head-label">Compensation Evaluations</span>
      <div className="metric-grid">
        <MetricCell
          title="Avg $/hr"
          value={idle ? "—" : formatRate(avg, imputed)}
          data={series(rates)}
          active={!idle}
        />
        <MetricCell
          title="Remote"
          value={idle ? "—" : `${remotePct}%`}
          data={series(idle ? [] : results.map((r) => (r.remote ? 100 : 0)))}
          active={!idle}
        />
        <MetricCell
          title="Top $/hr"
          value={idle ? "—" : formatRate(top, imputed)}
          data={series(idle ? [] : [...rates].sort((a, b) => a - b))}
          active={!idle}
        />
        <MetricCell
          title="Results"
          value={idle ? "0" : String(results.length)}
          data={series(idle ? [] : rates.map((_, i) => i + 1))}
          active={!idle}
        />
      </div>
    </section>
  );
}
