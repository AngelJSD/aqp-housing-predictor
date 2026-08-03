const MLFLOW_URL = process.env.MLFLOW_URL ?? "http://localhost:5001";

type ModelVersion = { version: string; run_id: string; status: string };

async function getLatestVersion(registeredModelName: string): Promise<ModelVersion> {
  const res = await fetch(`${MLFLOW_URL}/api/2.0/mlflow/registered-models/get-latest-versions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: registeredModelName }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`MLflow lookup failed for ${registeredModelName}: ${res.status}`);
  const { model_versions } = await res.json();
  return model_versions[0];
}

async function getRunMetrics(runId: string): Promise<Record<string, number>> {
  const res = await fetch(`${MLFLOW_URL}/api/2.0/mlflow/runs/get?run_id=${runId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`MLflow run lookup failed for ${runId}: ${res.status}`);
  const { run } = await res.json();
  return Object.fromEntries(run.data.metrics.map((m: { key: string; value: number }) => [m.key, m.value]));
}

export type ModelSummary = {
  registeredModelName: string;
  version: string;
  metrics: Record<string, number>;
};

export async function getModelSummary(registeredModelName: string): Promise<ModelSummary> {
  const version = await getLatestVersion(registeredModelName);
  const metrics = await getRunMetrics(version.run_id);
  return { registeredModelName, version: version.version, metrics };
}
