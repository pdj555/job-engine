/** Downsample to at most `points` averages. Never pads with zeros. */
export function downsample(values: number[], points = 16): number[] {
  if (values.length === 0) return [];
  if (values.length <= points) return values;
  const out: number[] = [];
  const size = values.length / points;
  for (let i = 0; i < points; i++) {
    const start = Math.floor(i * size);
    const end = Math.max(Math.floor((i + 1) * size), start + 1);
    const slice = values.slice(start, end);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}
