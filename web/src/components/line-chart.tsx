"use client";

type LineChartProps = {
  data: number[];
  height?: number;
  active?: boolean;
};

export function LineChart({ data, height = 64, active = true }: LineChartProps) {
  const w = 220;
  const h = height;
  const padL = 28;
  const padR = 4;
  const padT = 4;
  const padB = 4;

  const series = data.length >= 2 ? data : data.length === 1 ? [data[0], data[0]] : [0, 0];
  const max = Math.max(...series);
  const min = Math.min(...series);
  const range = max - min;
  const idle = !active || series.every((v) => v === 0);

  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const points = series
    .map((v, i) => {
      const x = padL + (i / (series.length - 1)) * plotW;
      const y =
        range === 0
          ? padT + plotH * 0.4
          : padT + (1 - (v - min) / range) * plotH;
      return `${x},${y}`;
    })
    .join(" ");

  const ticks = 4;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full block" preserveAspectRatio="none" aria-hidden>
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const y = padT + (i / ticks) * plotH;
        const val = max - (i / ticks) * (range || 1);
        return (
          <g key={i}>
            <line
              x1={padL}
              y1={y}
              x2={w - padR}
              y2={y}
              stroke="var(--line)"
              strokeWidth="0.5"
              opacity={idle ? 0.08 : 0.14}
            />
            {!idle && range > 0 && (
              <text
                x={0}
                y={y + 3}
                fill="var(--muted)"
                fontSize="6"
                fontFamily="var(--font-mono), monospace"
              >
                {val >= 1000 ? `${Math.round(val / 1000)}k` : val.toFixed(val < 10 ? 1 : 0)}
              </text>
            )}
          </g>
        );
      })}
      {idle ? (
        <line
          x1={padL}
          y1={padT + plotH}
          x2={w - padR}
          y2={padT + plotH}
          stroke="var(--faint)"
          strokeWidth="1"
          strokeDasharray="2 3"
        />
      ) : (
        <polyline fill="none" stroke="var(--accent)" strokeWidth="1.4" points={points} />
      )}
    </svg>
  );
}
