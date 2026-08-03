"use client";

import { useState } from "react";

type PredictResult = {
  predicted_price_usd: number;
  operation_type: string;
  model_version: string;
  latency_ms: number;
};

// Venta has real, higher variability across CV folds than Alquiler (task
// 3: CV std=0.34 on mean R2=0.64, vs. Alquiler std=0.12 on mean 0.66) — a
// qualitative flag, not a numeric interval (that would need quantile
// regression, not built anywhere in this pipeline). See task 9's plan
// entry for the full reasoning.
const CONFIDENCE_NOTE: Record<string, string> = {
  Venta: "El modelo de Venta tiene más variabilidad entre distintos conjuntos de validación — trata esta estimación como una referencia orientativa, no una tasación exacta.",
  Alquiler: "El modelo de Alquiler es comparativamente más estable, pero sigue siendo una estimación, no una tasación exacta.",
};

export default function PredictorPage() {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    const form = new FormData(e.currentTarget);
    const body = {
      district: form.get("district"),
      surface: Number(form.get("surface")),
      property_type: form.get("property_type"),
      operation_type: form.get("operation_type"),
    };
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.message ?? "Error al predecir");
      } else {
        setResult(data);
      }
    } catch {
      setError("No se pudo contactar la API de inferencia.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6 max-w-lg">
      <h1 className="text-2xl font-semibold">¿El precio es razonable?</h1>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1 text-sm">
          Distrito
          <input name="district" required className="rounded border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900" placeholder="ej. Cayma" />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Superficie (m²)
          <input name="surface" type="number" min="1" step="any" required className="rounded border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900" />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Tipo de propiedad
          <input name="property_type" required className="rounded border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900" placeholder="ej. Departamento" />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Tipo de operación
          <select name="operation_type" required className="rounded border border-zinc-300 px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900">
            <option value="Venta">Venta</option>
            <option value="Alquiler">Alquiler</option>
          </select>
        </label>
        <button type="submit" disabled={loading} className="rounded-full bg-zinc-950 px-5 py-2 text-white disabled:opacity-50 dark:bg-zinc-50 dark:text-black">
          {loading ? "Calculando..." : "Estimar precio"}
        </button>
      </form>

      {error && <p className="text-red-600">{error}</p>}

      {result && (
        <div className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
          <p className="text-3xl font-semibold">
            ${result.predicted_price_usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}
            {result.operation_type === "Alquiler" && <span className="text-base font-normal text-zinc-500"> /mes</span>}
          </p>
          <p className="mt-2 text-sm text-zinc-500">{CONFIDENCE_NOTE[result.operation_type]}</p>
          <p className="mt-2 text-xs text-zinc-400">
            Modelo v{result.model_version} · {result.latency_ms.toFixed(1)}ms
          </p>
        </div>
      )}
    </div>
  );
}
