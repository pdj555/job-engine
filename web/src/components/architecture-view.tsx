import { Panel } from "./panel";

const STACK: [string, string][] = [
  ["brain", "OpenAI Agents SDK — plans web searches"],
  ["transport", "search_web tool · Engine fallback"],
  ["ranking", "deterministic · Opportunity.score()"],
  ["surfaces", "CLI · API · Web"],
];

export function ArchitectureView() {
  return (
    <>
      <Panel label="Architecture">
        <div className="flow">
          <span className="flow-end">goal</span>
          <span className="flow-arrow" aria-hidden>
            ↓
          </span>

          <div className="flow-node">
            <span className="flow-zone">autonomous</span>
            <span className="flow-title">OpenAI Agents SDK</span>
            <span className="flow-sub">plan · research the open web · extract</span>
          </div>

          <span className="flow-edge" aria-hidden>
            ↓ searches + opportunities
          </span>

          <div className="flow-node">
            <span className="flow-zone">deterministic</span>
            <span className="flow-title">
              rank · <b>$/hour</b>
            </span>
            <span className="flow-sub">posted pay only · office −30% · thin listings sink</span>
          </div>

          <span className="flow-arrow" aria-hidden>
            ↓
          </span>
          <span className="flow-end">CLI · API · Web</span>
        </div>

        <p className="about mt-3">
          The brain decides <em>what</em> to surface. The deterministic core owns{" "}
          <em>the $/hour</em> — it never invents a number it is graded on.
        </p>
      </Panel>

      <Panel label="Stack">
        <div className="metric-grid">
          {STACK.map(([key, value]) => (
            <div className="metric-cell" key={key}>
              <div className="metric-title">{key}</div>
              <div className="flow-sub">{value}</div>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}
