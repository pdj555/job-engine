import type { AgentResponse, SearchResponse } from "./types";

export type HealthResponse = {
  status: string;
  search_ready: boolean;
  agent_ready: boolean;
  apis: {
    openai: boolean;
    brave: boolean;
    perplexity: boolean;
  };
};

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const data = (await res.json()) as { error?: string; detail?: string };
    return data.error ?? data.detail ?? fallback;
  } catch {
    const text = await res.text();
    return text || fallback;
  }
}

export async function searchJobs(query: string, limit = 20): Promise<SearchResponse> {
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=${limit}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await readError(res, `Search failed (${res.status})`));
  return res.json();
}

export async function agentSearch(query: string, limit = 20): Promise<AgentResponse> {
  const res = await fetch("/api/agent", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ q: query, limit }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await readError(res, `Agent failed (${res.status})`));
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
