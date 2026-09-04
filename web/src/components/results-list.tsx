"use client";

import type { Opportunity } from "@/lib/types";
import { formatPayRange, formatRate } from "@/lib/format";
import { Panel } from "./panel";

export function ResultsList({
  results,
  onAdd,
}: {
  results: Opportunity[];
  onAdd: (opp: Opportunity) => void;
}) {
  const maxRate = Math.max(...results.map((r) => r.refined_rate ?? r.dollars_per_hour ?? 0), 1);

  if (results.length === 0) {
    return (
      <Panel label="Search Results">
        <p className="hint py-6 text-center">Run a search to see ranked results</p>
      </Panel>
    );
  }

  return (
    <Panel label="Search Results">
      <ul className="result-list">
        {results.map((opp, i) => {
          const rate = opp.refined_rate ?? opp.dollars_per_hour ?? 0;
          const pct = maxRate > 0 ? (rate / maxRate) * 100 : 0;

          return (
            <li key={opp.url} className="result-row">
              <div className="flex gap-2 items-start min-w-0">
                <span className="result-rank">{String(i + 1).padStart(2, "0")}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-col sm:flex-row sm:justify-between gap-2">
                    <div className="min-w-0">
                      <a
                        href={opp.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block leading-snug break-words hover:opacity-75"
                      >
                        {opp.title}
                      </a>
                      <p className="hint mt-1 truncate">
                        {opp.company ?? "—"} · {opp.remote ? "remote" : "onsite"} ·{" "}
                        {opp.pay != null || opp.pay_low != null || opp.pay_high != null
                          ? `${formatPayRange(opp.pay_low, opp.pay_high ?? opp.pay)}/yr`
                          : "—"}
                      </p>
                    </div>
                    <div className="flex sm:flex-col items-center sm:items-end gap-2 shrink-0">
                      <span className="stat-value">
                        {formatRate(opp.refined_rate ?? opp.dollars_per_hour, opp.rate_imputed)}
                      </span>
                      <button type="button" onClick={() => onAdd(opp)} className="btn">
                        + pipeline
                      </button>
                    </div>
                  </div>
                  <div className="bar-track mt-2" aria-hidden>
                    <div className="bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}
