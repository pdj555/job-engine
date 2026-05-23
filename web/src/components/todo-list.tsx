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
      <div className="flex justify-between items-baseline mb-2">
        <span className="stat-label">Completion</span>
        <span className="text-[9px] tabular-nums text-[var(--fg)]">{completionPct}%</span>
      </div>
      <div className="bar-track mb-5">
        <div className="bar-fill" style={{ width: `${completionPct}%` }} />
      </div>

      <div className="flex items-center justify-between mb-4 gap-3">
        <div className="flex gap-0.5">
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
        <span className="text-[8px] text-[var(--accent-muted)] tabular-nums tracking-wide shrink-0">
          {activeCount} active · {doneCount} done
        </span>
      </div>

      <div
        className={`flex gap-2.5 mb-2 border border-[var(--border)] px-3 py-2.5 bg-[var(--bg)] ${
          flash ? "flash" : ""
        }`}
      >
        <span className="text-[var(--text-subtle)] select-none pt-px text-[11px]">[ ]</span>
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
          className="flex-1 bg-transparent placeholder:text-[var(--text-subtle)] focus:outline-none text-[12px]"
          autoComplete="off"
        />
      </div>
      <p className="text-[8px] text-[var(--accent-muted)] mb-4 tracking-wide">
        <kbd>n</kbd> new · Enter save · Esc clear
      </p>

      {filtered.length === 0 ? (
        <div className="py-10 text-center border border-dashed border-[var(--border)] bg-[var(--bg)]">
          <p className="text-[9px] uppercase tracking-[0.12em] text-[var(--accent-muted)]">
            {filter === "active" ? "No active tasks" : "Nothing here"}
          </p>
          <p className="text-[8px] text-[var(--text-subtle)] mt-2">
            Add above or save from search results
          </p>
        </div>
      ) : (
        <ul className="scroll max-h-[380px] overflow-y-auto divide-soft">
          {filtered.map((todo, i) => (
            <li
              key={todo.id}
              className={`flex items-start gap-2.5 py-3 group fade-up ${
                todo.done ? "opacity-40" : ""
              }`}
              style={{ animationDelay: `${i * 15}ms` }}
            >
              <button
                type="button"
                onClick={() => toggle(todo.id)}
                className="text-[var(--accent-muted)] hover:text-[var(--fg)] cursor-pointer tabular-nums w-5 text-left shrink-0 text-[11px] transition-colors"
                aria-label={todo.done ? "Mark incomplete" : "Mark complete"}
              >
                {todo.done ? "[x]" : "[ ]"}
              </button>
              <div className="flex-1 min-w-0">
                <span
                  className={`block leading-snug text-[12px] ${todo.done ? "line-through" : ""}`}
                >
                  {todo.text}
                </span>
                {todo.opportunityUrl && (
                  <a
                    href={todo.opportunityUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-[8px] text-[var(--accent-muted)] truncate hover:text-[var(--fg)] mt-1 transition-colors"
                  >
                    {todo.opportunityUrl.replace(/^https?:\/\//, "")}
                  </a>
                )}
              </div>
              <button
                type="button"
                onClick={() => remove(todo.id)}
                className="text-[8px] uppercase tracking-[0.12em] text-[var(--text-subtle)] group-hover:text-[var(--accent-muted)] hover:!text-[var(--fg)] cursor-pointer shrink-0 transition-colors"
                aria-label="Delete"
              >
                del
              </button>
            </li>
          ))}
        </ul>
      )}

      {doneCount > 0 && (
        <button type="button" onClick={clearDone} className="btn btn-ghost w-full mt-4">
          clear completed
        </button>
      )}
    </Panel>
  );
}
