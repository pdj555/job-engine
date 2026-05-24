import type { Todo } from "./types";

const KEY = "job-engine-todos";

// null = not loaded yet (server + first client render). A non-null array is the source of truth.
let cache: Todo[] | null = null;
const listeners = new Set<() => void>();

export function subscribe(listener: () => void): () => void {
  if (cache === null) {
    try {
      const raw = localStorage.getItem(KEY);
      cache = raw ? (JSON.parse(raw) as Todo[]) : [];
    } catch {
      cache = [];
    }
  }
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getSnapshot(): Todo[] | null {
  return cache;
}

export function getServerSnapshot(): Todo[] | null {
  return null;
}

export function mutateTodos(updater: (prev: Todo[]) => Todo[]): void {
  cache = updater(cache ?? []);
  try {
    localStorage.setItem(KEY, JSON.stringify(cache));
  } catch {
    // ignore write failures (quota, private mode)
  }
  for (const listener of listeners) listener();
}

export function createTodo(text: string, opportunityUrl?: string): Todo {
  return {
    id: crypto.randomUUID(),
    text: text.trim(),
    done: false,
    createdAt: Date.now(),
    opportunityUrl,
  };
}
