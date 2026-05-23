"use client";

function path(values: number[], w: number, h: number, pad: number) {
  if (values.length === 0) return "";
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const step = (w - pad * 2) / Math.max(values.length - 1, 1);

  return values
    .map((v, i) => {
      const x = pad + i * step;
      const y = h - pad - ((v - min) / range) * (h - pad * 2);
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");
}

export function LineChart({
  values,
  title,
  format = (v) => String(Math.round(v)),
}: {
  values: number[];
  title: string;
  format?: (v: number) => string;
}) {
  if (values.length === 0) return null;

  const w = 240;
  const h = 72;
  const pad = 4;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const d = path(values, w, h, pad);
  const last = values[values.length - 1];

  return (
    <div className="panel-grid border border-[var(--accent-soft)] p-3">
      <p className="text-[10px] uppercase tracking-widest mb-2">
        {title}: <span className="font-bold">{format(last)}</span>
      </p>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[72px]" aria-hidden>
        {[0.25, 0.5, 0.75].map((p) => (
          <line
            key={p}
            x1={pad}
            y1={h * p}
            x2={w - pad}
            y2={h * p}
            stroke="var(--accent)"
            strokeWidth="0.3"
            opacity="0.15"
          />
        ))}
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.2" />
        {values.map((v, i) => {
          const step = (w - pad * 2) / Math.max(values.length - 1, 1);
          const x = pad + i * step;
          const range = max - min || 1;
          const y = h - pad - ((v - min) / range) * (h - pad * 2);
          return <circle key={i} cx={x} cy={y} r="1.5" fill="var(--accent)" />;
        })}
      </svg>
      <div className="flex justify-between mt-1 text-[8px] tabular-nums text-[var(--accent-muted)]">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>
    </div>
  );
}
