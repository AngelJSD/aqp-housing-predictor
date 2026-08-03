const MLFLOW_TRACKING_URI = process.env.MLFLOW_TRACKING_URI ?? "http://localhost:5001";

// Resolves the real registered version from MLflow's Model Registry (task
// 6) instead of inventing an arbitrary version string — cached at startup,
// not re-fetched per request.
export async function getLatestModelVersion(registeredModelName) {
  const res = await fetch(`${MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get-latest-versions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: registeredModelName }),
  });
  if (!res.ok) {
    throw new Error(`MLflow registry lookup failed for ${registeredModelName}: ${res.status} ${await res.text()}`);
  }
  const { model_versions } = await res.json();
  if (!model_versions?.length) {
    throw new Error(`No registered versions found for ${registeredModelName}`);
  }
  return model_versions[0].version;
}
