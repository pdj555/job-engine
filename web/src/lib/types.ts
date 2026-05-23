export type Opportunity = {
  title: string;
  company: string | null;
  url: string;
  pay: number | null;
  hours_per_week: number | null;
  dollars_per_hour: number | null;
  remote: boolean;
  score: number;
};

export type SearchResponse = {
  results: Opportunity[];
  count: number;
};

export type Todo = {
  id: string;
  text: string;
  done: boolean;
  createdAt: number;
  opportunityUrl?: string;
};

export type TodoFilter = "all" | "active" | "done";
