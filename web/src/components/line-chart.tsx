"use client";

type LineChartProps = {
  data: number[];
  height?: number;
};

export function LineChart({ data, height = 64 }: LineChartProps) {
  const w = 220;
  const h = height;
  const padL = 28;
  const padR = 4;
  const padT = 4;
  const padB = 4;

  const series = data.length >= 2 ? data : [0, 0];
  const max = Math.max(...series);
  const min = Math.min(...series);
  const range = max - min || 1;

  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const points = series
    .map((v, i) => {
      const x = padL + (i / (series.length - 1)) * plotW;
      const y = padT + (1 - (v - min) / range) * plotH;
      return `${x},${y}`;
    })
    .join(" ");

  const ticks = 4;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full block" preserveAspectRatio="none" aria-hidden>
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const y = padT + (i / ticks) * plotH;
        const val = max - (i / ticks) * range;
        return (
          <g key={i}>
            <line
              x1={padL}
              y1={y}
              x2={w - padR}
              y2={y}
              stroke="var(--accent)"
              strokeWidth="0.5"
              opacity="0.25"
            />
            <text
              x={0}
              y={y + 3}
              fill="var(--accent)"
              fontSize="6"
              opacity="0.55"
              fontFamily="var(--font-mono), monospace"
            >
              {val >= 1000 ? `${Math.round(val / 1000)}k` : val.toFixed(val < 10 ? 1 : 0)}
            </text>
          </g>
        );
      })}
      <polyline fill="none" stroke="var(--accent-bright)" strokeWidth="1.2" points={points} />
    </svg>
  );
}
