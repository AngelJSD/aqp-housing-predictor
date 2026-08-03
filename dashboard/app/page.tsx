import Link from "next/link";

const SECTIONS = [
  { href: "/predictor", title: "Predictor", description: "¿El precio que me piden es razonable? Estimación en vivo vía la API de inferencia." },
  { href: "/models", title: "Modelos", description: "Versiones registradas en MLflow y sus métricas de entrenamiento." },
  { href: "/monitoring", title: "Monitoreo", description: "Latencia, throughput y tasa de error de la API, vía Prometheus." },
  { href: "/drift", title: "Drift", description: "Reportes de Evidently: features de entrenamiento vs. tráfico reciente." },
  { href: "/features", title: "Catálogo de features", description: "Documentación humano-legible de cada feature del modelo." },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold">Predictor de precios — Arequipa</h1>
        <p className="mt-2 text-zinc-500">Dashboard de observabilidad del pipeline de MLOps.</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {SECTIONS.map((s) => (
          <Link key={s.href} href={s.href} className="rounded-lg border border-zinc-200 p-4 hover:border-zinc-400 dark:border-zinc-800 dark:hover:border-zinc-600">
            <h2 className="font-medium">{s.title}</h2>
            <p className="mt-1 text-sm text-zinc-500">{s.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
