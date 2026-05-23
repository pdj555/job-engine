"use client";

import { useEffect, useState } from "react";
import type { Opportunity } from "@/lib/types";
import { checkHealth } from "@/lib/api";
import { useTodos } from "@/lib/use-todos";
import { Header } from "./header";
import { Hero } from "./hero";
import { SearchPanel } from "./search-panel";
import { TodoList } from "./todo-list";
import { StatsPanel } from "./stats-panel";
import { ThemeProvider } from "./theme-provider";

export function JobEngineApp() {
  const [results, setResults] = useState<Opportunity[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const todos = useTodos();

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
    <ThemeProvider>
      <Header apiOnline={apiOnline} />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-10 sm:py-14 space-y-12 flex-1 w-full">
        <Hero
          resultCount={results.length}
          pipelinePct={todos.completionPct}
          apiOnline={apiOnline}
        />

        <div className="grid lg:grid-cols-[1fr_260px] gap-8 lg:gap-10">
          <div className="space-y-6 min-w-0">
            <SearchPanel onResults={setResults} onAdd={handleAdd} />
            <TodoList {...todos} />
          </div>
          <StatsPanel
            results={results}
            apiOnline={apiOnline}
            activeCount={todos.activeCount}
            doneCount={todos.doneCount}
            completionPct={todos.completionPct}
          />
        </div>
      </main>

      <footer className="border-t border-[var(--border)] py-5 text-[8px] uppercase tracking-[0.16em] text-[var(--text-subtle)] text-center">
        ranked by effective $/hr · MIT
      </footer>
    </ThemeProvider>
  );
}
