export type PaySource = "posted" | "schema" | "ats" | null;

export type Opportunity = {
  title: string;
  company: string | null;
  url: string;
  pay: number | null;
  hours_per_week: number | null;
  dollars_per_hour: number | null;
  refined_rate: number | null;
  rate_imputed: boolean;
  remote: boolean;
  score: number;
  pay_source: PaySource;
  hours_source: "posted" | "schema" | "ats" | null;
};

export type SearchResponse = {
  results: Opportunity[];
  count: number;
};

export type AgentResponse = SearchResponse & {
  searches: string[];
};

export type Todo = {
  id: string;
  text: string;
  done: boolean;
  createdAt: number;
  opportunityUrl?: string;
};

export type TodoFilter = "all" | "active" | "done";
