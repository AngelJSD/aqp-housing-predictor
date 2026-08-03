const API_URL = process.env.API_URL ?? "http://localhost:3000";

// Proxies server-side to the inference API (internal Docker service name
// in production, host-mapped port in local dev) — the browser only ever
// talks to this dashboard's own origin, never directly to the `api`
// service. See proyecto-mlops-plan.md, task 9, for why.
export async function POST(request: Request) {
  const body = await request.json();
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
