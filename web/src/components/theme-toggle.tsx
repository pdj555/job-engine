"use client";

import { useTheme } from "./theme-provider";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();

  return (
    <button
      type="button"
      onClick={toggle}
      className="cursor-pointer uppercase tracking-widest text-[10px] text-[var(--accent-muted)] hover:text-[var(--fg)] transition-colors"
      aria-label="Toggle theme"
    >
      {theme === "light" ? "dark mode" : "light mode"}
    </button>
  );
}
