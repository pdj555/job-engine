"use client";

import { useEffect, useState } from "react";
import type { Opportunity } from "@/lib/types";
import { checkHealth } from "@/lib/api";
import { useTodos } from "@/lib/use-todos";
import { AgentTrace } from "./agent-trace";
import { FrameRail } from "./frame-rail";
import { Hero } from "./hero";
import { MetricsSection } from "./metrics-section";
import { ResultsList } from "./results-list";
import { SearchComposer } from "./search-composer";
import { TodoList } from "./todo-list";

export function JobEngineApp() {
  const [results, setResults] = useState<Opportunity[]>([]);
  const [agentTrace, setAgentTrace] = useState<string[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [searching, setSearching] = useState(false);
  const todos = useTodos();

  const live = searching || results.length > 0 || apiOnline === true;
  const status = apiOnline === null ? "…" : apiOnline ? "online" : "offline";

  useEffect(() => {
    const poll = () =>
      checkHealth().then((health) => {
        setApiOnline(health !== null);
      });

    poll();
    const id = setInterval(poll, 15000);
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
      <FrameRail status={status} resultCount={results.length} />

      <main className="frame-body">
        <Hero
          resultCount={results.length}
          pipelinePct={todos.completionPct}
          live={live}
        />

        <SearchComposer
          onResults={setResults}
          onSearching={setSearching}
          onTrace={setAgentTrace}
          apiOnline={apiOnline}
        />

        <AgentTrace searches={agentTrace} />

        <MetricsSection results={results} />

        <div className="split">
          <ResultsList results={results} onAdd={handleAdd} />
          <TodoList {...todos} />
        </div>
      </main>
    </div>
  );
}
