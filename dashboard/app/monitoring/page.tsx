import { queryInstant } from "@/lib/prometheus";
import { RequestsChart } from "./RequestsChart";

function labelFor(metric: Record<string, string>): string {
  return `${metric.method} ${metric.route} (${metric.status_code})`;
}

export default async function MonitoringPage() {
  const [requestCounts, avgLatency] = await Promise.all([
    queryInstant("http_requests_total"),
    queryInstant("http_request_duration_seconds_sum / http_request_duration_seconds_count"),
  ]);

  const latencyByLabel = new Map(avgLatency.map((r) => [labelFor(r.metric), Number(r.value[1])]));

  const rows = requestCounts.map((r) => ({
    label: labelFor(r.metric),
    requests: Number(r.value[1]),
    avgLatencyMs: (latencyByLabel.get(labelFor(r.metric)) ?? 0) * 1000,
  }));

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">Monitoreo (Prometheus)</h1>
      {rows.length === 0 ? (
        <p className="text-zinc-500">Sin tráfico registrado todavía.</p>
      ) : (
        <>
          <RequestsChart rows={rows} />
          <table className="text-sm">
            <thead>
              <tr className="text-left text-zinc-500">
                <th className="pr-6">Ruta</th>
                <th className="pr-6">Requests</th>
                <th>Latencia promedio (ms)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label}>
                  <td className="pr-6 font-mono">{row.label}</td>
                  <td className="pr-6">{row.requests}</td>
                  <td>{row.avgLatencyMs.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
