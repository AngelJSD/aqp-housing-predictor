import { getModelSummary } from "@/lib/mlflow";

const REGISTERED_MODELS = ["arequipa-price-venta", "arequipa-price-alquiler"];

export default async function ModelsPage() {
  const summaries = await Promise.all(REGISTERED_MODELS.map(getModelSummary));

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">Modelos registrados (MLflow)</h1>
      <div className="grid gap-6 sm:grid-cols-2">
        {summaries.map((summary) => (
          <div key={summary.registeredModelName} className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
            <h2 className="font-mono text-sm text-zinc-500">{summary.registeredModelName}</h2>
            <p className="mb-3 text-lg font-semibold">Versión {summary.version}</p>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              {Object.entries(summary.metrics).map(([key, value]) => (
                <div key={key} className="contents">
                  <dt className="text-zinc-500">{key}</dt>
                  <dd className="text-right font-mono">{value.toFixed(4)}</dd>
                </div>
              ))}
            </dl>
          </div>
        ))}
      </div>
    </div>
  );
}
