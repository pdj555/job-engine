"use client";

import { useCallback, useEffect, useState } from "react";
import type { Todo } from "./types";
import { createTodo, loadTodos, saveTodos } from "./todo-storage";

export function useTodos() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [ready, setReady] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    setTodos(loadTodos());
    setReady(true);
  }, []);

  useEffect(() => {
    if (ready) saveTodos(todos);
  }, [todos, ready]);

  const add = useCallback((text: string, opportunityUrl?: string) => {
    const trimmed = text.trim();
    if (!trimmed) return false;
    setTodos((prev) => [createTodo(trimmed, opportunityUrl), ...prev]);
    setTick((t) => t + 1);
    return true;
  }, []);

  const toggle = useCallback((id: string) => {
    setTodos((prev) =>
      prev.map((t) => (t.id === id ? { ...t, done: !t.done } : t))
    );
  }, []);

  const remove = useCallback((id: string) => {
    setTodos((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const clearDone = useCallback(() => {
    setTodos((prev) => prev.filter((t) => !t.done));
  }, []);

  const activeCount = todos.filter((t) => !t.done).length;
  const doneCount = todos.filter((t) => t.done).length;
  const completionPct =
    todos.length > 0 ? Math.round((doneCount / todos.length) * 100) : 0;

  return {
    todos,
    ready,
    tick,
    add,
    toggle,
    remove,
    clearDone,
    activeCount,
    doneCount,
    completionPct,
  };
}
