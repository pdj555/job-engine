import type { SearchResponse } from "./types";

export async function searchJobs(query: string, limit = 20): Promise<SearchResponse> {
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=${limit}`, {
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Search failed (${res.status})`);
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
