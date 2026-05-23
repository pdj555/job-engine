"use client";

import { useTheme } from "./theme-provider";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();

  return (
    <button
      type="button"
      onClick={toggle}
      className="btn btn-ghost !px-2 !py-1"
      aria-label="Toggle theme"
    >
      {theme === "light" ? "dark" : "light"}
    </button>
  );
}
