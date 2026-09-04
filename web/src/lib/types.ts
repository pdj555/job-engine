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
  remote: boolean;
  score: number;
};

export type SearchResponse = {
  results: Opportunity[];
  count: number;
};

// The agent response adds the autonomous trace — the queries the agent chose to run.
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
