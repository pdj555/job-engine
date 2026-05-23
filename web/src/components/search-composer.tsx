"use client";

import { useEffect, useRef, useState } from "react";
import { searchJobs } from "@/lib/api";
import { Panel } from "./panel";

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
    <Panel label="Query">
      <form onSubmit={runSearch}>
        <div className="input-row">
          <span className="prompt" aria-hidden>
            &gt;
          </span>
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="roles, contracts, grants…"
            autoComplete="off"
            spellCheck={false}
            aria-label="Search query"
            disabled={apiOnline === false}
          />
          <button type="submit" disabled={disabled} className="btn shrink-0">
            {loading ? "…" : "run"}
          </button>
        </div>
      </form>

      <p className="hint mt-2">
        {apiOnline === false
          ? "api offline · run `job-engine serve` on :8000"
          : "/ focus · enter to search"}
      </p>

      {loading && <p className="loading py-4">loading...</p>}

      {empty && !loading && !error && (
        <p className="hint py-2">no results · try different keywords</p>
      )}

      {error && <p className="hint py-2">{error}</p>}
    </Panel>
  );
}
