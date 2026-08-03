import { readFile } from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-node";

const MODELS_DIR = process.env.MODELS_DIR ?? path.resolve(import.meta.dirname, "../../data/processed/models");

const OPERATION_TYPES = ["venta", "alquiler"];

// Loaded once at startup, keyed by lowercase operation_type — matches the
// file naming ml/training/export_onnx.py already uses ({op}_xgb.onnx,
// {op}_categories.json), so there's one source of truth for that naming.
export async function loadModels(modelsDir = MODELS_DIR) {
  const models = {};
  for (const op of OPERATION_TYPES) {
    const session = await ort.InferenceSession.create(path.join(modelsDir, `${op}_xgb.onnx`));
    const categories = JSON.parse(await readFile(path.join(modelsDir, `${op}_categories.json`), "utf-8"));
    models[op] = { session, categories };
  }
  return models;
}
