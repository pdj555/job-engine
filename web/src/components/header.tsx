"use client";

export function Header({
  apiOnline,
  resultCount,
}: {
  apiOnline: boolean | null;
  resultCount: number;
}) {
  const status = apiOnline === null ? "…" : apiOnline ? "online" : "offline";

  return (
    <header className="header-bar grid grid-cols-[1fr_auto_1fr] items-center gap-4">
      <div className="meta truncate">
        API <span className="text-[var(--accent)]">{status}</span>
        <span className="hidden sm:inline mx-2 opacity-30">·</span>
        <span className="hidden sm:inline">
          {resultCount > 0 ? `${resultCount} ranked` : "idle"}
        </span>
      </div>

      <span className="text-[11px] sm:text-xs uppercase tracking-[0.28em] text-[var(--fg)] font-medium">
        Job Engine
      </span>

      <div className="flex justify-end">
        <a
          href="https://github.com/pdj555/job-engine"
          target="_blank"
          rel="noopener noreferrer"
          className="meta hover:text-[var(--accent)] transition-colors"
        >
          github
        </a>
      </div>
    </header>
  );
}
