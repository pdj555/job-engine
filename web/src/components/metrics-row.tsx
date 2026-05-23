"use client";

import type { Opportunity } from "@/lib/types";
import { formatRate } from "@/lib/format";

function spark(values: number[], buckets = 12): number[] {
  if (values.length === 0) return Array(buckets).fill(0);
  const out: number[] = [];
  const size = Math.max(1, Math.ceil(values.length / buckets));
  for (let i = 0; i < buckets; i++) {
    const slice = values.slice(i * size, (i + 1) * size);
    out.push(slice.length ? Math.max(...slice) : 0);
  }
  return out;
}

function MetricCard({
  label,
  value,
  bars,
}: {
  label: string;
  value: string;
  bars: number[];
}) {
  const max = Math.max(...bars, 1);

  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="sparkline" aria-hidden>
        {bars.map((v, i) => (
          <div
            key={i}
            className="spark-bar"
            style={{ height: `${Math.max(8, (v / max) * 100)}%` }}
          />
        ))}
      </div>
    </div>
  );
}

export function MetricsRow({ results }: { results: Opportunity[] }) {
  const rates = results
    .map((r) => r.dollars_per_hour ?? 0)
    .filter((r) => r > 0);
  const avgRate =
    rates.length > 0 ? rates.reduce((a, b) => a + b, 0) / rates.length : null;
  const remotePct =
    results.length > 0
      ? Math.round((results.filter((r) => r.remote).length / results.length) * 100)
      : 0;
  const topRate = rates.length > 0 ? Math.max(...rates) : 0;

  if (results.length === 0) return null;

  return (
    <section>
      <p className="meta mb-4">Compensation Metrics</p>
      <div className="metric-grid">
        <MetricCard
          label="Avg $/hr"
          value={formatRate(avgRate)}
          bars={spark(rates)}
        />
        <MetricCard
          label="Remote %"
          value={`${remotePct}%`}
          bars={spark(
            results.map((r) => (r.remote ? 100 : 0)),
            12
          )}
        />
        <MetricCard
          label="Top $/hr"
          value={formatRate(topRate)}
          bars={spark([...rates].sort((a, b) => a - b))}
        />
      </div>
    </section>
  );
}
