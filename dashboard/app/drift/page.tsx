const OPERATIONS = ["venta", "alquiler"];

export default function DriftPage() {
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">Reportes de drift (Evidently)</h1>
      <p className="text-sm text-zinc-500">
        Features de entrada (distrito/superficie/tipo de propiedad) — entrenamiento vs. tráfico reciente. Ver{" "}
        <code>ml/monitoring/drift_report.py</code>.
      </p>
      {OPERATIONS.map((op) => (
        <div key={op} className="flex flex-col gap-2">
          <h2 className="text-lg font-medium capitalize">{op}</h2>
          <iframe src={`/api/drift/${op}`} className="h-[600px] w-full rounded-lg border border-zinc-200 dark:border-zinc-800" />
        </div>
      ))}
    </div>
  );
}
