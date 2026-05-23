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
  tick: number;
  add: (text: string, url?: string) => boolean;
  toggle: (id: string) => void;
  remove: (id: string) => void;
  clearDone: () => void;
  activeCount: number;
  doneCount: number;
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
      <div className="flex justify-between meta mb-3">
        <span>{todos.length} tasks</span>
        <span>{completionPct}%</span>
      </div>
      <div className="bar-track mb-4">
        <div className="bar-fill" style={{ width: `${completionPct}%` }} />
      </div>

      <div className="flex gap-1 mb-4" role="tablist" aria-label="Filter tasks">
        {FILTERS.map((f) => (
          <button
            key={f}
            type="button"
            role="tab"
            aria-selected={filter === f}
            onClick={() => setFilter(f)}
            className={`btn btn-sm btn-ghost meta ${filter === f ? "btn-ghost-active" : ""}`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="input-shell mb-4">
        <span className="input-prompt" aria-hidden>
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
          className="input"
          autoComplete="off"
          aria-label="New task"
        />
      </div>

      {filtered.length === 0 ? (
        <p className="hint py-6 text-center">No tasks</p>
      ) : (
        <ul className="scroll max-h-[min(320px,40vh)] overflow-y-auto divide-soft">
          {filtered.map((todo) => (
            <li
              key={todo.id}
              className={`flex items-start gap-2 py-3 min-w-0 ${todo.done ? "opacity-40" : ""}`}
            >
              <button
                type="button"
                onClick={() => toggle(todo.id)}
                className="min-w-[44px] min-h-[44px] -ml-2 flex items-center justify-center text-[var(--accent-muted)] hover:text-[var(--accent)] cursor-pointer text-[11px] shrink-0 meta normal-case tracking-normal"
                aria-label={todo.done ? "Mark incomplete" : "Mark complete"}
              >
                {todo.done ? "[x]" : "[ ]"}
              </button>
              <div className="flex-1 min-w-0 pt-2">
                <span
                  className={`block text-[15px] leading-snug break-words ${todo.done ? "line-through text-muted" : ""}`}
                >
                  {todo.text}
                </span>
                {todo.opportunityUrl && (
                  <a
                    href={todo.opportunityUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block hint mt-1 truncate hover:text-[var(--accent)]"
                  >
                    {todo.opportunityUrl.replace(/^https?:\/\//, "")}
                  </a>
                )}
              </div>
              <button
                type="button"
                onClick={() => remove(todo.id)}
                className="btn btn-sm btn-ghost shrink-0 mt-1 meta"
                aria-label="Remove"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {todos.some((t) => t.done) && (
        <button type="button" onClick={clearDone} className="btn btn-sm btn-ghost w-full mt-4 meta">
          clear done
        </button>
      )}
    </Panel>
  );
}
