"use client";

type LineChartProps = {
  data: number[];
  height?: number;
};

export function LineChart({ data, height = 52 }: LineChartProps) {
  const w = 200;
  const h = height;
  const pad = 4;
  const series = data.length >= 2 ? data : [0, 0];

  const max = Math.max(...series);
  const min = Math.min(...series);
  const range = max - min || 1;

  const points = series
    .map((v, i) => {
      const x = pad + (i / (series.length - 1)) * (w - pad * 2);
      const y = h - pad - ((v - min) / range) * (h - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");

  const ticks = 4;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full block"
      preserveAspectRatio="none"
      aria-hidden
    >
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const y = pad + (i / ticks) * (h - pad * 2);
        return (
          <line
            key={i}
            x1={pad}
            y1={y}
            x2={w - pad}
            y2={y}
            stroke="var(--accent)"
            strokeWidth="0.5"
            opacity="0.2"
          />
        );
      })}
      <polyline
        fill="none"
        stroke="var(--accent)"
        strokeWidth="1.25"
        points={points}
      />
    </svg>
  );
}
