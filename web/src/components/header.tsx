import { ThemeToggle } from "./theme-toggle";

export function Header() {
  return (
    <header className="border-b border-[var(--border)] px-4 sm:px-6 py-3 flex items-center justify-between bg-[var(--surface)] sticky top-0 z-10">
      <div className="flex items-center gap-4 sm:gap-6 text-[10px] uppercase tracking-widest">
        <span className="text-[var(--accent-muted)] hidden sm:inline">v0.2.0 run</span>
        <ThemeToggle />
      </div>
      <span className="text-[10px] uppercase tracking-[0.2em]">Job Engine</span>
      <a
        href="https://github.com/pdj555/job-engine"
        target="_blank"
        rel="noopener noreferrer"
        className="text-[10px] uppercase tracking-widest text-[var(--accent-muted)] hover:text-[var(--fg)] transition-colors"
      >
        github
      </a>
    </header>
  );
}
