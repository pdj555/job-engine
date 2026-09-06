"use client";

import { useState } from "react";
import type { Opportunity } from "@/lib/types";
import { formatPay, formatRate } from "@/lib/format";
import { Panel } from "./panel";

type Filter = "all" | "listed";

export function ResultsList({
  results,
  onAdd,
}: {
  results: Opportunity[];
  onAdd: (opp: Opportunity) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const shown = filter === "listed" ? results.filter((r) => r.pay_source === "posted") : results;
  const maxScore = Math.max(...shown.map((r) => r.score), 1);

  if (results.length === 0) {
    return (
      <Panel label="Search Results">
        <p className="hint py-6 text-center">Run a search to see ranked results</p>
      </Panel>
    );
  }

  return (
    <Panel label="Search Results">
      <div className="flex gap-1 mb-3">
        {(["all", "listed"] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`btn btn-ghost ${filter === f ? "btn-ghost-active" : ""}`}
            aria-pressed={filter === f}
          >
            {f}
          </button>
        ))}
      </div>
      {shown.length === 0 ? (
        <p className="hint py-6 text-center">No listings with stated pay</p>
      ) : (
        <ul className="result-list">
          {shown.map((opp, i) => {
            const pct = maxScore > 0 ? (opp.score / maxScore) * 100 : 0;
            const hours = opp.hours_per_week ? `${opp.hours_per_week}h/wk` : "hours ?";
            const pay = `${formatPay(opp.pay)}/yr${opp.pay_source === "posted" ? " listed" : ""}`;

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
                          {opp.company ?? "—"} · {opp.remote ? "remote" : "onsite"} · {pay} · {hours}
                        </p>
                      </div>
                      <div className="flex sm:flex-col items-center sm:items-end gap-2 shrink-0">
                        <span className="stat-value">{formatRate(opp.dollars_per_hour)}</span>
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
      )}
    </Panel>
  );
}
