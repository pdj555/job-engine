"use client";

import { useSyncExternalStore } from "react";

const KEY = "job-engine-theme";

// The `dark` class on <html> is the source of truth (set pre-hydration by the
// inline script in layout.tsx). Read it directly instead of mirroring it in state.
const listeners = new Set<() => void>();

function isDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function setDark(next: boolean): void {
  document.documentElement.classList.toggle("dark", next);
  try {
    localStorage.setItem(KEY, next ? "dark" : "light");
  } catch {
    // ignore write failures (quota, private mode)
  }
  for (const listener of listeners) listener();
}

export function ThemeToggle() {
  const dark = useSyncExternalStore(subscribe, isDark, () => false);

  return (
    <button
      type="button"
      className="btn-theme"
      onClick={() => setDark(!isDark())}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? "light" : "dark"}
    </button>
  );
}
