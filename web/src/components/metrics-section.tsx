"use client";

import type { Opportunity } from "@/lib/types";
import { formatRate } from "@/lib/format";
import { downsample } from "@/lib/series";
import { LineChart } from "./line-chart";

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
          value={idle ? "—" : formatRate(avg)}
          data={downsample(rates)}
          active={!idle}
        />
        <MetricCell
          title="Remote"
          value={idle ? "—" : `${remotePct}%`}
          data={downsample(idle ? [] : results.map((r) => (r.remote ? 100 : 0)))}
          active={!idle}
        />
        <MetricCell
          title="Top $/hr"
          value={idle ? "—" : formatRate(top)}
          data={downsample(idle ? [] : [...rates].sort((a, b) => a - b))}
          active={!idle}
        />
        <MetricCell
          title="Results"
          value={idle ? "0" : String(results.length)}
          data={downsample(idle ? [] : results.map((_, i) => i + 1))}
          active={!idle}
        />
      </div>
    </section>
  );
}
