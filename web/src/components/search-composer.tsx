"use client";

import { useEffect, useRef, useState } from "react";
import { searchJobs } from "@/lib/api";

export function SearchComposer({
  onResults,
  onSearching,
  apiOnline,
  searchReady,
}: {
  onResults: (results: import("@/lib/types").Opportunity[]) => void;
  onSearching?: (loading: boolean) => void;
  apiOnline: boolean | null;
  searchReady: boolean;
}) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [empty, setEmpty] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const disabled =
    loading || !query.trim() || apiOnline === false || (apiOnline === true && !searchReady);

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
    if (!q || apiOnline === false) return;

    setLoading(true);
    onSearching?.(true);
    setError(null);
    setEmpty(false);

    try {
      const data = await searchJobs(q);
      onResults(data.results);
      if (data.results.length === 0) setEmpty(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      onResults([]);
    } finally {
      setLoading(false);
      onSearching?.(false);
    }
  }

  return (
    <div className="search-zone">
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
          disabled={apiOnline === false}
        />
        <button type="submit" disabled={disabled} className="composer-btn">
          {loading ? "…" : "Search"}
        </button>
      </form>
      <p className="composer-hint">
        {apiOnline === false
          ? "Start the API with `job-engine serve` on :8000"
          : "Press / to focus"}
      </p>

      {loading && <p className="loading">loading...</p>}

      {empty && !loading && !error && (
        <p className="hint text-center" style={{ maxWidth: "36rem" }}>
          No results for that query. Try different keywords.
        </p>
      )}

      {error && (
        <section className="panel w-full" style={{ maxWidth: "36rem" }}>
          <span className="panel-label">Error</span>
          <p style={{ fontFamily: "var(--font-sans)", color: "var(--accent-dim)" }}>{error}</p>
        </section>
      )}
    </div>
  );
}
