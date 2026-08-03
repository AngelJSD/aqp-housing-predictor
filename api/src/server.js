import "dotenv/config";
import Fastify from "fastify";
import * as ort from "onnxruntime-node";

import { loadModels } from "./models.js";
import { encodeInput } from "./encode.js";
import { getLatestModelVersion } from "./mlflow.js";
import { logPrediction } from "./db.js";
import { register, httpRequestDuration, httpRequestsTotal } from "./metrics.js";

const REGISTERED_MODEL_NAMES = { venta: "arequipa-price-venta", alquiler: "arequipa-price-alquiler" };

const fastify = Fastify({ logger: true });

const models = await loadModels();
const modelVersions = Object.fromEntries(
  await Promise.all(
    Object.entries(REGISTERED_MODEL_NAMES).map(async ([op, name]) => [op, await getLatestModelVersion(name)])
  )
);
fastify.log.info({ modelVersions }, "Loaded models and resolved registry versions");

const predictSchema = {
  body: {
    type: "object",
    required: ["district", "surface", "property_type", "operation_type"],
    properties: {
      district: { type: "string", minLength: 1 },
      surface: { type: "number", exclusiveMinimum: 0 },
      property_type: { type: "string", minLength: 1 },
      operation_type: { type: "string", enum: ["Venta", "Alquiler"] },
    },
  },
};

fastify.addHook("onResponse", (request, reply, done) => {
  const labels = {
    method: request.method,
    route: request.routeOptions?.url ?? "unknown",
    status_code: reply.statusCode,
  };
  httpRequestsTotal.inc(labels);
  httpRequestDuration.observe(labels, reply.elapsedTime / 1000);
  done();
});

fastify.get("/health", async () => ({ status: "ok" }));

fastify.get("/metrics", async (request, reply) => {
  reply.header("Content-Type", register.contentType);
  return register.metrics();
});

fastify.post("/predict", { schema: predictSchema }, async (request, reply) => {
  const start = performance.now();
  const { district, surface, property_type, operation_type } = request.body;
  const op = operation_type.toLowerCase();
  const { session, categories } = models[op];

  const input = encodeInput({ district, surface, property_type }, categories);
  const tensor = new ort.Tensor("float32", input, [1, input.length]);
  const output = await session.run({ input: tensor });
  const predictedPriceUsd = Math.exp(output.variable.data[0]);
  const latencyMs = performance.now() - start;

  await logPrediction({
    operationType: operation_type,
    district,
    surface,
    propertyType: property_type,
    predictedPriceUsd,
    modelVersion: modelVersions[op],
    latencyMs,
  });

  return {
    predicted_price_usd: predictedPriceUsd,
    operation_type,
    model_version: modelVersions[op],
    latency_ms: latencyMs,
  };
});

const port = Number(process.env.PORT ?? 3000);
await fastify.listen({ host: "0.0.0.0", port });
