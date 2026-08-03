const PROMETHEUS_URL = process.env.PROMETHEUS_URL ?? "http://localhost:9090";

type InstantResult = { metric: Record<string, string>; value: [number, string] };

export async function queryInstant(query: string): Promise<InstantResult[]> {
  const url = `${PROMETHEUS_URL}/api/v1/query?query=${encodeURIComponent(query)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Prometheus query failed: ${res.status}`);
  const { data } = await res.json();
  return data.result;
}
