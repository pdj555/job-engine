"use client";

import { useEffect, useRef, useState } from "react";
import { searchJobs } from "@/lib/api";

export function SearchComposer({
  onResults,
  onSearching,
}: {
  onResults: (results: import("@/lib/types").Opportunity[]) => void;
  onSearching?: (loading: boolean) => void;
}) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
      onResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      onResults([]);
    } finally {
      setLoading(false);
      onSearching?.(false);
    }
  }

  return (
    <>
      <div className="composer-wrap">
        <form onSubmit={runSearch} className="composer">
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search roles, contracts, grants…"
            className="composer-input"
            autoComplete="off"
            spellCheck={false}
            aria-label="Search query"
          />
          <button type="submit" disabled={loading || !query.trim()} className="composer-btn">
            {loading ? "…" : "Run"}
          </button>
        </form>
      </div>

      {loading && <p className="loading">loading…</p>}

      {error && (
        <section className="panel">
          <span className="panel-label">Error</span>
          <p style={{ color: "var(--text)", lineHeight: 1.5 }}>{error}</p>
        </section>
      )}
    </>
  );
}
