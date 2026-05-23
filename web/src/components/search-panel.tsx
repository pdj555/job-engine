"use client";

import { useEffect, useRef, useState } from "react";
import type { Opportunity } from "@/lib/types";
import { searchJobs } from "@/lib/api";
import { formatPay, formatRate } from "@/lib/format";
import { MetricsRow } from "./metrics-row";
import { Panel } from "./panel";

export function SearchPanel({
  onResults,
  onAdd,
  onSearching,
}: {
  onResults: (results: Opportunity[]) => void;
  onAdd: (opp: Opportunity) => void;
  onSearching?: (loading: boolean) => void;
}) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Opportunity[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const maxRate = Math.max(...results.map((r) => r.dollars_per_hour ?? 0), 1);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function runSearch(e?: React.FormEvent) {
    e?.preventDefault();
    const q = query.trim();
    if (!q) return;

    setLoading(true);
    onSearching?.(true);
    setError(null);

    try {
      const data = await searchJobs(q);
      setResults(data.results);
      onResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults([]);
      onResults([]);
    } finally {
      setLoading(false);
      onSearching?.(false);
    }
  }

  return (
    <div className="stack min-w-0">
      <form onSubmit={runSearch} className="composer">
        <span className="composer-prompt" aria-hidden>
          &gt;
        </span>
        <input
          ref={inputRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="AI engineer, contract, grant…"
          className="composer-input"
          autoComplete="off"
          spellCheck={false}
          aria-label="Search query"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="composer-btn shrink-0"
        >
          {loading ? "…" : "Run"}
        </button>
      </form>

      {loading && <p className="loading-text">loading…</p>}

      {error && (
        <Panel label="Error">
          <p className="leading-relaxed break-words">{error}</p>
        </Panel>
      )}

      {!loading && results.length > 0 && (
        <>
          <MetricsRow results={results} />

          <Panel label={`${results.length} results`}>
            <ul className="scroll max-h-[min(480px,55vh)] overflow-y-auto divide-soft">
              {results.map((opp, i) => {
                const rate = opp.dollars_per_hour ?? 0;
                const pct = maxRate > 0 ? (rate / maxRate) * 100 : 0;

                return (
                  <li key={opp.url} className="py-3 first:pt-0 last:pb-0 result-row">
                    <div className="flex gap-3 items-start min-w-0">
                      <span className="result-rank">{String(i + 1).padStart(2, "0")}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-col sm:flex-row sm:justify-between gap-2 sm:gap-4">
                          <div className="min-w-0">
                            <a
                              href={opp.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block leading-snug hover:text-[var(--accent)] break-words transition-colors"
                            >
                              {opp.title}
                            </a>
                            <p className="hint mt-1.5 truncate">
                              {opp.company ?? "—"} · {opp.remote ? "remote" : "onsite"} ·{" "}
                              {formatPay(opp.pay)}/yr
                            </p>
                          </div>
                          <div className="flex sm:flex-col items-center sm:items-end gap-2 shrink-0">
                            <span className="stat-value">{formatRate(opp.dollars_per_hour)}</span>
                            <button
                              type="button"
                              onClick={() => onAdd(opp)}
                              className="btn btn-sm btn-ghost"
                            >
                              + pipeline
                            </button>
                          </div>
                        </div>
                        <div className="bar-track mt-3" aria-hidden>
                          <div className="bar-fill" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          </Panel>
        </>
      )}
    </div>
  );
}
