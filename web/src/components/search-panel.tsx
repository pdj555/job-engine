"use client";

import { useEffect, useRef, useState } from "react";
import type { Opportunity } from "@/lib/types";
import { searchJobs } from "@/lib/api";
import { formatPay, formatRate } from "@/lib/format";
import { Panel } from "./panel";

export function SearchPanel({
  onResults,
  onAdd,
}: {
  onResults: (results: Opportunity[]) => void;
  onAdd: (opp: Opportunity) => void;
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
    }
  }

  return (
    <div className="space-y-4">
      <Panel label="Search Query">
        <form onSubmit={runSearch} className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="AI engineer, contract, grant..."
            className="flex-1 bg-[var(--bg)] border border-[var(--accent-soft)] px-3 py-2.5 placeholder:text-[var(--accent-muted)] focus:border-[var(--accent)] transition-colors"
            autoComplete="off"
            spellCheck={false}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="border border-[var(--border)] px-4 py-2.5 uppercase text-[10px] tracking-widest cursor-pointer disabled:opacity-30 hover:bg-[var(--accent)] hover:text-[var(--bg)] transition-colors"
          >
            {loading ? "..." : "run"}
          </button>
        </form>
        <p className="mt-2 text-[10px] text-[var(--accent-muted)]">
          <kbd>/</kbd> to focus · ranked by $/hr
        </p>
      </Panel>

      {loading && (
        <p className="py-12 text-center text-[var(--accent-muted)] lowercase">loading...</p>
      )}

      {error && (
        <Panel label="Error">
          <p>{error}</p>
          <p className="mt-2 text-[10px] text-[var(--accent-muted)]">
            Run <code>job-engine serve</code> first.
          </p>
        </Panel>
      )}

      {!loading && results.length > 0 && (
        <Panel label={`Results · ${results.length}`}>
          <ul className="scroll max-h-[480px] overflow-y-auto divide-y divide-[var(--accent-soft)]">
            {results.map((opp, i) => {
              const rate = opp.dollars_per_hour ?? 0;
              const pct = maxRate > 0 ? (rate / maxRate) * 100 : 0;

              return (
                <li
                  key={opp.url}
                  className="py-3 first:pt-0 last:pb-0 fade-up group"
                  style={{ animationDelay: `${i * 25}ms` }}
                >
                  <div className="flex gap-3">
                    <span className="text-[10px] text-[var(--accent-muted)] tabular-nums pt-0.5 w-5">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between gap-3">
                        <div className="min-w-0">
                          <a
                            href={opp.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block truncate hover:underline"
                          >
                            {opp.title}
                          </a>
                          <p className="text-[10px] text-[var(--accent-muted)] mt-1 truncate">
                            {opp.company ?? "Unknown"} · {opp.remote ? "remote" : "onsite"} ·{" "}
                            {formatPay(opp.pay)}/yr · {opp.hours_per_week ?? "?"}h/wk
                          </p>
                        </div>
                        <div className="text-right shrink-0">
                          <p className="stat-value text-base">{formatRate(opp.dollars_per_hour)}</p>
                          <button
                            type="button"
                            onClick={() => onAdd(opp)}
                            className="mt-1 text-[9px] uppercase tracking-widest text-[var(--accent-muted)] opacity-0 group-hover:opacity-100 hover:text-[var(--fg)] cursor-pointer transition-opacity"
                          >
                            + pipeline
                          </button>
                        </div>
                      </div>
                      <div className="bar-track mt-2">
                        <div className="bar-fill" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        </Panel>
      )}
    </div>
  );
}
