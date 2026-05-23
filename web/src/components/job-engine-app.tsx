"use client";

import { useEffect, useState } from "react";
import type { Opportunity } from "@/lib/types";
import { checkHealth } from "@/lib/api";
import { useTodos } from "@/lib/use-todos";
import { Header } from "./header";
import { Hero } from "./hero";
import { SearchPanel } from "./search-panel";
import { TodoList } from "./todo-list";

export function JobEngineApp() {
  const [results, setResults] = useState<Opportunity[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [searching, setSearching] = useState(false);
  const todos = useTodos();

  const live = searching || results.length > 0 || apiOnline === true;

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
    <div className="page">
      <Header apiOnline={apiOnline} resultCount={results.length} />

      <main className="page-main">
        <div className="stack">
          <Hero pipelinePct={todos.completionPct} live={live} />

          <div className="dashboard">
            <div className="min-w-0">
              <SearchPanel
                onResults={setResults}
                onAdd={handleAdd}
                onSearching={setSearching}
              />
            </div>

            <aside id="pipeline" className="min-w-0 scroll-mt-6 lg:sticky lg:top-[4.25rem]">
              <TodoList {...todos} />
            </aside>
          </div>
        </div>
      </main>

      <footer className="site-footer">Ranked by effective $/hr</footer>
    </div>
  );
}
