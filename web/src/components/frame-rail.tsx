"use client";

import { ThemeToggle } from "./theme-toggle";

export function FrameRail({
  status,
  resultCount,
}: {
  status: string;
  resultCount: number;
}) {
  return (
    <header className="frame-rail">
      <span className="truncate">
        ranked by $/hr
        {resultCount > 0 ? ` · ${resultCount} results` : ""}
        <span className="mx-2">·</span>
        <ThemeToggle />
      </span>
      <span className="frame-rail-center">Job Engine</span>
      <span className="frame-rail-right">
        <a href="https://github.com/pdj555/job-engine" target="_blank" rel="noopener noreferrer">
          github
        </a>
        <span className="hidden sm:inline opacity-60">api {status}</span>
      </span>
    </header>
  );
}
