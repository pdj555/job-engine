"use client";

import { useCallback, useSyncExternalStore } from "react";
import type { Todo } from "./types";
import {
  createTodo,
  getServerSnapshot,
  getSnapshot,
  mutateTodos,
  subscribe,
} from "./todo-storage";

const EMPTY: Todo[] = [];

export function useTodos() {
  const stored = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const ready = stored !== null;
  const todos = stored ?? EMPTY;

  const add = useCallback((text: string, opportunityUrl?: string) => {
    const trimmed = text.trim();
    if (!trimmed) return false;
    mutateTodos((prev) => [createTodo(trimmed, opportunityUrl), ...prev]);
    return true;
  }, []);

  const toggle = useCallback((id: string) => {
    mutateTodos((prev) =>
      prev.map((t) => (t.id === id ? { ...t, done: !t.done } : t))
    );
  }, []);

  const remove = useCallback((id: string) => {
    mutateTodos((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const clearDone = useCallback(() => {
    mutateTodos((prev) => prev.filter((t) => !t.done));
  }, []);

  const doneCount = todos.filter((t) => t.done).length;
  const completionPct =
    todos.length > 0 ? Math.round((doneCount / todos.length) * 100) : 0;

  return {
    todos,
    ready,
    add,
    toggle,
    remove,
    clearDone,
    completionPct,
  };
}
