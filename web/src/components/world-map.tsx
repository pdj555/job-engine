"use client";

const NODES = [
  { x: 24, y: 36, hot: true },
  { x: 42, y: 28, hot: false },
  { x: 52, y: 42, hot: true },
  { x: 68, y: 30, hot: false },
  { x: 78, y: 38, hot: true },
  { x: 48, y: 58, hot: false },
  { x: 62, y: 62, hot: true },
  { x: 32, y: 65, hot: false },
];
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6], [6, 7], [5, 0],
];

export function WorldMap({ live }: { live: boolean }) {
  return (
    <div className="relative aspect-[5/3] min-h-[100px]">
      <svg viewBox="0 0 100 70" className="absolute inset-0 w-full h-full" aria-hidden>
        <ellipse cx="28" cy="38" rx="14" ry="16" fill="none" stroke="var(--accent)" strokeWidth="0.4" opacity="0.35" />
        <ellipse cx="52" cy="34" rx="11" ry="13" fill="none" stroke="var(--accent)" strokeWidth="0.4" opacity="0.35" />
        <ellipse cx="72" cy="36" rx="16" ry="14" fill="none" stroke="var(--accent)" strokeWidth="0.4" opacity="0.35" />
        <ellipse cx="58" cy="58" rx="10" ry="8" fill="none" stroke="var(--accent)" strokeWidth="0.4" opacity="0.35" />

        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            x1={NODES[a].x}
            y1={NODES[a].y}
            x2={NODES[b].x}
            y2={NODES[b].y}
            stroke="var(--accent)"
            strokeWidth="0.35"
            opacity={live ? 0.45 : 0.15}
          />
        ))}

        {NODES.map((n, i) => (
          <circle
            key={i}
            cx={n.x}
            cy={n.y}
            r="1.8"
            fill={n.hot && live ? "var(--node)" : "var(--accent)"}
            opacity={live ? 1 : 0.25}
          />
        ))}
      </svg>
      <span
        className="absolute bottom-1 left-2 uppercase tracking-widest"
        style={{ fontSize: "8px", color: "var(--accent-dim)" }}
      >
        {live ? "live" : "idle"}
      </span>
    </div>
  );
}
