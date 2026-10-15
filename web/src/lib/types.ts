export type PaySource = "posted" | "schema" | "ats" | "snippet" | "unverified" | null;

export type Opportunity = {
  title: string;
  company: string | null;
  url: string;
  pay: number | null;
  pay_low: number | null;
  pay_high: number | null;
  hours_per_week: number | null;
  dollars_per_hour: number | null;
  refined_rate: number | null;
  rate_imputed: boolean;
  remote: boolean | null;
  score: number;
  pay_source: PaySource;
  pay_source_url: string | null;
  pay_is_annualized: boolean;
  hours_source: PaySource;
  remote_source: PaySource;
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
