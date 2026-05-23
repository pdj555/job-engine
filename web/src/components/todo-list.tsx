"use client";

import { useEffect, useRef, useState } from "react";
import type { Todo, TodoFilter } from "@/lib/types";
import { Panel } from "./panel";

const FILTERS: TodoFilter[] = ["all", "active", "done"];

export function TodoList({
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
  const [flash, setFlash] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const prevTick = useRef(tick);

  useEffect(() => {
    if (tick !== prevTick.current) {
      prevTick.current = tick;
      setFlash(true);
      const id = setTimeout(() => setFlash(false), 500);
      return () => clearTimeout(id);
    }
  }, [tick]);

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
    <Panel label="Pipeline Tasks">
      <div className="flex justify-between mb-2">
        <span className="stat-label">Completion</span>
        <span className="text-[10px] tabular-nums">{completionPct}%</span>
      </div>
      <div className="bar-track mb-4">
        <div className="bar-fill" style={{ width: `${completionPct}%` }} />
      </div>

      <div className="flex items-center justify-between mb-3">
        <div className="flex gap-1">
          {FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`px-2.5 py-1 text-[9px] uppercase tracking-widest cursor-pointer border transition-colors ${
                filter === f
                  ? "border-[var(--border)] bg-[var(--accent)] text-[var(--bg)]"
                  : "border-transparent text-[var(--accent-muted)] hover:text-[var(--fg)]"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <span className="text-[9px] text-[var(--accent-muted)] tabular-nums">
          {activeCount} active · {doneCount} done
        </span>
      </div>

      <div
        className={`flex gap-2 mb-3 border border-[var(--accent-soft)] px-2 py-2 bg-[var(--bg)] ${
          flash ? "flash" : ""
        }`}
      >
        <span className="text-[var(--accent-muted)] select-none pt-px">[ ]</span>
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
          placeholder="Apply, follow up, research..."
          className="flex-1 bg-transparent placeholder:text-[var(--accent-muted)] focus:outline-none"
          autoComplete="off"
        />
      </div>
      <p className="text-[9px] text-[var(--accent-muted)] mb-3 -mt-1">
        <kbd>n</kbd> new task · Enter save · Esc clear
      </p>

      {filtered.length === 0 ? (
        <div className="py-8 text-center border border-dashed border-[var(--accent-soft)]">
          <p className="text-[10px] uppercase tracking-widest text-[var(--accent-muted)]">
            {filter === "active" ? "No active tasks" : "Nothing here"}
          </p>
          <p className="text-[9px] text-[var(--accent-muted)] mt-1 opacity-70">
            Add above or save from search results
          </p>
        </div>
      ) : (
        <ul className="scroll max-h-[360px] overflow-y-auto">
          {filtered.map((todo, i) => (
            <li
              key={todo.id}
              className={`flex items-start gap-2 py-2.5 border-b border-[var(--accent-soft)] last:border-0 group fade-up ${
                todo.done ? "opacity-45" : ""
              }`}
              style={{ animationDelay: `${i * 20}ms` }}
            >
              <button
                type="button"
                onClick={() => toggle(todo.id)}
                className="text-[var(--accent-muted)] hover:text-[var(--fg)] cursor-pointer tabular-nums w-5 text-left shrink-0"
                aria-label={todo.done ? "Mark incomplete" : "Mark complete"}
              >
                {todo.done ? "[x]" : "[ ]"}
              </button>
              <div className="flex-1 min-w-0">
                <span className={`block leading-snug ${todo.done ? "line-through" : ""}`}>
                  {todo.text}
                </span>
                {todo.opportunityUrl && (
                  <a
                    href={todo.opportunityUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-[9px] text-[var(--accent-muted)] truncate hover:underline mt-0.5"
                  >
                    {todo.opportunityUrl.replace(/^https?:\/\//, "")}
                  </a>
                )}
              </div>
              <button
                type="button"
                onClick={() => remove(todo.id)}
                className="text-[9px] uppercase tracking-widest text-[var(--accent-muted)] opacity-0 group-hover:opacity-100 hover:text-[var(--fg)] cursor-pointer shrink-0"
                aria-label="Delete"
              >
                del
              </button>
            </li>
          ))}
        </ul>
      )}

      {doneCount > 0 && (
        <button
          type="button"
          onClick={clearDone}
          className="mt-3 w-full border border-[var(--accent-soft)] py-1.5 text-[9px] uppercase tracking-widest text-[var(--accent-muted)] hover:border-[var(--border)] hover:text-[var(--fg)] cursor-pointer transition-colors"
        >
          clear completed
        </button>
      )}
    </Panel>
  );
}
