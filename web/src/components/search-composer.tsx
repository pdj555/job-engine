"use client";

import { useEffect, useRef, useState } from "react";
import { agentSearch, searchJobs } from "@/lib/api";
import type { Opportunity } from "@/lib/types";
import { Panel } from "./panel";

type Mode = "search" | "agent";
const MODES: Mode[] = ["search", "agent"];

export function SearchComposer({
  onResults,
  onSearching,
  onTrace,
  apiOnline,
  agentReady,
}: {
  onResults: (results: Opportunity[]) => void;
  onSearching?: (loading: boolean) => void;
  onTrace?: (searches: string[]) => void;
  apiOnline: boolean | null;
  agentReady: boolean;
}) {
  const [mode, setMode] = useState<Mode>("search");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [empty, setEmpty] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const disabled = loading || !query.trim();

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

  function switchMode(next: Mode) {
    if (next === mode) return;
    setMode(next);
    setError(null);
    setEmpty(false);
    onTrace?.([]); // a prior agent trace doesn't belong to the other mode
  }

  async function runSearch(e?: React.FormEvent) {
    e?.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    onSearching?.(true);
    setError(null);
    setEmpty(false);

    try {
      if (mode === "agent") {
        const data = await agentSearch(q);
        onResults(data.results);
        onTrace?.(data.searches);
        if (data.results.length === 0) setEmpty(true);
      } else {
        const data = await searchJobs(q);
        onResults(data.results);
        onTrace?.([]);
        if (data.results.length === 0) setEmpty(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      onResults([]);
    } finally {
      setLoading(false);
      onSearching?.(false);
    }
  }

  const hint =
    apiOnline === false
      ? "api offline · run `job-engine serve` on :8000"
      : mode === "agent"
        ? agentReady
          ? "autonomous · plans its own searches"
          : "open-web fallback · no OPENAI_API_KEY"
        : "/ focus · enter to search";

  return (
    <Panel label="Query">
      <div className="flex gap-1 mb-3">
        {MODES.map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => switchMode(m)}
            className={`btn btn-ghost ${mode === m ? "btn-ghost-active" : ""}`}
            aria-pressed={mode === m}
          >
            {m}
          </button>
        ))}
      </div>

      <form onSubmit={runSearch}>
        <div className="input-row">
          <span className="prompt" aria-hidden>
            &gt;
          </span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={mode === "agent" ? "give the agent a goal…" : "roles, contracts, grants…"}
            autoComplete="off"
            spellCheck={false}
            aria-label="Search query"
          />
          <button type="submit" disabled={disabled} className="btn shrink-0">
            {loading ? "…" : mode === "agent" ? "dispatch" : "run"}
          </button>
        </div>
      </form>

      <p className="hint mt-2">{hint}</p>

      {loading && (
        <p className="loading py-4">{mode === "agent" ? "agent working" : "loading"}...</p>
      )}

      {empty && !loading && !error && (
        <p className="hint py-2">no results · try different keywords</p>
      )}

      {error && <p className="hint py-2">{error}</p>}
    </Panel>
  );
}
