"use client";

import { useEffect, useState } from "react";
import type { Opportunity } from "@/lib/types";
import { checkHealth } from "@/lib/api";
import { useTodos } from "@/lib/use-todos";
import { Hero } from "./hero";
import { MetricsSection } from "./metrics-section";
import { ResultsList } from "./results-list";
import { SearchComposer } from "./search-composer";
import { TodoList } from "./todo-list";

export function JobEngineApp() {
  const [results, setResults] = useState<Opportunity[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [searching, setSearching] = useState(false);
  const todos = useTodos();

  const live = searching || results.length > 0 || apiOnline === true;
  const status = apiOnline === null ? "…" : apiOnline ? "online" : "offline";

  useEffect(() => {
    checkHealth().then(setApiOnline);
    const id = setInterval(() => checkHealth().then(setApiOnline), 15000);
    return () => clearInterval(id);
  }, []);

  function handleAdd(opp: Opportunity) {
    todos.add(
      `Apply: ${opp.title}${opp.company ? ` @ ${opp.company}` : ""}`,
      opp.url
    );
  }

  return (
    <div className="frame">
      <header className="frame-rail">
        <span className="truncate">
          api {status}
          {results.length > 0 ? ` · ${results.length} ranked` : ""}
        </span>
        <span>Job Engine</span>
        <span className="text-right">
          <a href="https://github.com/pdj555/job-engine" target="_blank" rel="noopener noreferrer">
            github
          </a>
        </span>
      </header>

      <main className="frame-body">
        <Hero
          resultCount={results.length}
          pipelinePct={todos.completionPct}
          live={live}
        />

        <SearchComposer
          onResults={setResults}
          onSearching={setSearching}
        />

        <MetricsSection results={results} />

        <div className="split">
          <ResultsList results={results} onAdd={handleAdd} />
          <TodoList {...todos} />
        </div>
      </main>
    </div>
  );
}
