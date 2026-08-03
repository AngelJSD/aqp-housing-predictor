import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextRequest } from "next/server";

const DRIFT_REPORTS_DIR = process.env.DRIFT_REPORTS_DIR ?? "../data/processed/drift_reports";
const VALID_OPERATIONS = ["venta", "alquiler"];

// Reads live from the mounted volume on every request (not a build-time
// copy) — drift_report.py can regenerate these between runs.
export async function GET(request: NextRequest, { params }: { params: Promise<{ operation: string }> }) {
  const { operation } = await params;
  if (!VALID_OPERATIONS.includes(operation)) {
    return new Response("Not found", { status: 404 });
  }
  try {
    const html = await readFile(path.join(DRIFT_REPORTS_DIR, `${operation}_drift.html`), "utf-8");
    return new Response(html, { headers: { "Content-Type": "text/html" } });
  } catch {
    return new Response("No hay reporte de drift generado todavía para este operation_type.", {
      status: 404,
      headers: { "Content-Type": "text/plain" },
    });
  }
}
