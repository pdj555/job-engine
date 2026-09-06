export type Opportunity = {
  title: string;
  company: string | null;
  url: string;
  pay: number | null;
  hours_per_week: number | null;
  dollars_per_hour: number | null;
  remote: boolean;
  score: number;
  pay_source: "posted" | null;
  hours_source: "posted" | null;
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
