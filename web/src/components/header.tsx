"use client";

import { ThemeToggle } from "./theme-toggle";

export function Header({ apiOnline }: { apiOnline: boolean | null }) {
  const status =
    apiOnline === null ? "checking" : apiOnline ? "online" : "offline";

  return (
    <header className="header-bar sticky top-0 z-20 px-4 sm:px-6 py-3.5 flex items-center justify-between">
      <div className="flex items-center gap-4 sm:gap-5 min-w-0">
        <span className="text-[9px] uppercase tracking-[0.16em] text-[var(--text-subtle)] hidden sm:inline">
          v0.2.0
        </span>
        <span className="chip">
          <span
            className={`chip-dot ${apiOnline ? "chip-dot-live" : ""}`}
            style={{ opacity: apiOnline === false ? 0.25 : 1 }}
          />
          api {status}
        </span>
        <ThemeToggle />
      </div>

      <span className="text-[10px] uppercase tracking-[0.18em] font-medium">
        Job Engine
      </span>

      <a
        href="https://github.com/pdj555/job-engine"
        target="_blank"
        rel="noopener noreferrer"
        className="text-[9px] uppercase tracking-[0.14em] text-[var(--accent-muted)] hover:text-[var(--fg)] transition-colors"
      >
        github
      </a>
    </header>
  );
}
