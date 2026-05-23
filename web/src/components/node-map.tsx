"use client";

const NODES = [
  [22, 35], [38, 22], [55, 40], [70, 25], [82, 38],
  [45, 58], [62, 65], [30, 68], [75, 55],
];
const EDGES = [[0,1],[1,2],[2,3],[3,4],[2,5],[5,6],[6,7],[6,8]];

export function NodeMap({ live }: { live: boolean }) {
  return (
    <div className="panel-grid aspect-[5/3] border border-[var(--accent-soft)] relative">
      <svg viewBox="0 0 100 70" className="absolute inset-0 w-full h-full" aria-hidden>
        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            x1={NODES[a][0]}
            y1={NODES[a][1]}
            x2={NODES[b][0]}
            y2={NODES[b][1]}
            stroke="var(--accent)"
            strokeWidth="0.35"
            opacity="0.35"
          />
        ))}
        {NODES.map(([x, y], i) => (
          <circle
            key={i}
            cx={x}
            cy={y}
            r="2"
            fill="var(--accent)"
            opacity={live ? 0.9 : 0.3}
            className={live ? "node-live" : undefined}
            style={{ animationDelay: `${i * 0.25}s` }}
          />
        ))}
      </svg>
      <span className="absolute bottom-2 left-2 text-[8px] uppercase tracking-widest text-[var(--accent-muted)]">
        {live ? "parallel fetch" : "standby"}
      </span>
    </div>
  );
}
