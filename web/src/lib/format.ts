export function formatRate(n: number | null, imputed = false): string {
  if (n == null) return "—";
  const val = `$${Math.round(n)}/hr`;
  return imputed ? `~${val}` : val;
}

export function formatPay(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${Math.round(n / 1_000)}k`;
  return `$${n}`;
}
