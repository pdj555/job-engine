"use client";

const NODES = [
  { x: 22, y: 38, hot: true },
  { x: 38, y: 28, hot: false },
  { x: 50, y: 42, hot: true },
  { x: 64, y: 30, hot: false },
  { x: 76, y: 36, hot: true },
  { x: 46, y: 56, hot: false },
  { x: 58, y: 62, hot: true },
  { x: 30, y: 64, hot: false },
  { x: 72, y: 54, hot: true },
];
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6], [6, 7], [5, 8], [4, 8],
];

export function WorldMap({ live }: { live: boolean }) {
  return (
    <div className="relative aspect-[5/3] min-h-[108px]">
      <svg viewBox="0 0 100 72" className="absolute inset-0 w-full h-full" aria-hidden>
        <path
          d="M18 38 Q28 28 38 32 T58 30 Q68 26 78 34 T82 42 Q72 48 62 44 T42 48 Q28 52 18 44 Z"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="0.5"
          opacity="0.35"
        />
        <path
          d="M44 52 Q52 48 58 50 T68 56 Q62 64 52 62 T38 58 Q42 54 44 52 Z"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="0.5"
          opacity="0.35"
        />

        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            x1={NODES[a].x}
            y1={NODES[a].y}
            x2={NODES[b].x}
            y2={NODES[b].y}
            stroke="var(--accent)"
            strokeWidth="0.35"
            opacity={live ? 0.5 : 0.18}
          />
        ))}

        {NODES.map((n, i) => (
          <circle
            key={i}
            cx={n.x}
            cy={n.y}
            r="1.7"
            fill={n.hot && live ? "var(--node)" : "var(--accent)"}
            opacity={live ? 1 : 0.3}
          />
        ))}
      </svg>
    </div>
  );
}
