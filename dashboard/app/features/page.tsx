import { getFeatureMetadata } from "@/lib/db";

// Without this, Next prerenders this page at build time (pg queries don't
// participate in the fetch-cache heuristic Next otherwise uses to detect
// dynamic pages) — confirmed for real: the built .html had live DB values
// baked in, would go stale until the next image rebuild.
export const dynamic = "force-dynamic";

export default async function FeaturesPage() {
  const features = await getFeatureMetadata();

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-semibold">Catálogo de features</h1>
      <div className="overflow-x-auto">
        <table className="w-full min-w-max text-left text-sm">
          <thead>
            <tr className="text-zinc-500">
              <th className="pr-4 py-2">Nombre</th>
              <th className="pr-4 py-2">Descripción</th>
              <th className="pr-4 py-2">Tipo</th>
              <th className="pr-4 py-2">Columna origen</th>
              <th className="pr-4 py-2">Feature view</th>
              <th className="pr-4 py-2">Owner</th>
              <th className="pr-4 py-2">Versión</th>
            </tr>
          </thead>
          <tbody>
            {features.map((f) => (
              <tr key={f.name} className="border-t border-zinc-200 dark:border-zinc-800">
                <td className="pr-4 py-2 font-mono">{f.name}</td>
                <td className="pr-4 py-2 max-w-md">{f.description}</td>
                <td className="pr-4 py-2 font-mono">{f.dtype}</td>
                <td className="pr-4 py-2 font-mono">{f.source_columns}</td>
                <td className="pr-4 py-2 font-mono">{f.feast_feature_view}</td>
                <td className="pr-4 py-2">{f.owner}</td>
                <td className="pr-4 py-2">{f.version}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
