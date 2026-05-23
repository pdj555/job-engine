import type { SearchResponse } from "./types";

export type HealthResponse = {
  status: string;
  search_ready: boolean;
  apis: {
    openai: boolean;
    brave: boolean;
    perplexity: boolean;
  };
};

export async function searchJobs(query: string, limit = 20): Promise<SearchResponse> {
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=${limit}`, {
    cache: "no-store",
  });

  if (!res.ok) {
    let message = `Search failed (${res.status})`;
    try {
      const data = (await res.json()) as { error?: string; detail?: string };
      message = data.error ?? data.detail ?? message;
    } catch {
      const text = await res.text();
      if (text) message = text;
    }
    throw new Error(message);
  }

  return res.json();
}

export async function checkHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
