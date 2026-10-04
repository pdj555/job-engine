export function formatRate(n: number | null, imputed = false): string {
  if (n == null) return "—";
  return `${imputed ? "~" : ""}$${Math.round(n)}/hr`;
}

export function formatPay(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${Math.round(n / 1_000)}k`;
  return `$${n}`;
}

export function formatPayRange(
  low: number | null,
  high: number | null,
  mid: number | null,
): string {
  if (low != null && high != null && low !== high) {
    return `${formatPay(low)}–${formatPay(high)}`;
  }
  return formatPay(mid ?? high ?? low);
}

export function isListed(source: string | null | undefined): boolean {
  return source === "posted" || source === "schema" || source === "ats";
}
