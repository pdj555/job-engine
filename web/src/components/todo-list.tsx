"use client";

import { useEffect, useRef, useState } from "react";
import type { Todo, TodoFilter } from "@/lib/types";
import { Panel } from "./panel";

const FILTERS: TodoFilter[] = ["all", "active", "done"];

export function TodoList({
  todos,
  ready,
  add,
  toggle,
  remove,
  clearDone,
  completionPct,
}: {
  todos: Todo[];
  ready: boolean;
  add: (text: string, url?: string) => boolean;
  toggle: (id: string) => void;
  remove: (id: string) => void;
  clearDone: () => void;
  completionPct: number;
}) {
  const [draft, setDraft] = useState("");
  const [filter, setFilter] = useState<TodoFilter>("active");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "n" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const filtered = todos.filter((t) => {
    if (filter === "active") return !t.done;
    if (filter === "done") return t.done;
    return true;
  });

  function submit() {
    if (add(draft)) setDraft("");
  }

  if (!ready) return null;

  return (
    <Panel label="Pipeline">
      <div className="meta-row">
        <span>{todos.length} tasks</span>
        <span>{completionPct}% complete</span>
      </div>
      <div className="bar-track mb-3">
        <div className="bar-fill" style={{ width: `${completionPct}%` }} />
      </div>

      <div className="flex gap-1 mb-3 flex-wrap">
        {FILTERS.map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`btn btn-ghost ${filter === f ? "btn-ghost-active" : ""}`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="input-row mb-3">
        <span className="prompt" aria-hidden>
          +
        </span>
        <input
          ref={inputRef}
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              submit();
            }
            if (e.key === "Escape") setDraft("");
          }}
          placeholder="Add task…"
          aria-label="New task"
        />
      </div>

      {filtered.length === 0 ? (
        <p className="hint py-4 text-center">No tasks</p>
      ) : (
        <ul className="todo-list">
          {filtered.map((todo) => (
            <li
              key={todo.id}
              className={`flex items-start gap-1 py-2 min-w-0 ${todo.done ? "opacity-40" : ""}`}
            >
              <button
                type="button"
                onClick={() => toggle(todo.id)}
                className="check-btn"
                aria-label={todo.done ? "Mark incomplete" : "Mark complete"}
              >
                {todo.done ? "[x]" : "[ ]"}
              </button>
              <div className="flex-1 min-w-0 pt-1">
                <span className={`block leading-snug break-words ${todo.done ? "line-through" : ""}`}>
                  {todo.text}
                </span>
                {todo.opportunityUrl && (
                  <a
                    href={todo.opportunityUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block hint mt-1 truncate hover:opacity-75"
                  >
                    {todo.opportunityUrl.replace(/^https?:\/\//, "")}
                  </a>
                )}
              </div>
              <button
                type="button"
                onClick={() => remove(todo.id)}
                className="btn btn-ghost shrink-0"
                aria-label="Remove"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {todos.some((t) => t.done) && (
        <button type="button" onClick={clearDone} className="btn w-full mt-3">
          clear done
        </button>
      )}
    </Panel>
  );
}
