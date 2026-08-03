import pg from "pg";

const pool = new pg.Pool({
  host: process.env.POSTGRES_HOST ?? "localhost",
  port: Number(process.env.POSTGRES_PORT ?? 5432),
  user: process.env.POSTGRES_USER,
  password: process.env.POSTGRES_PASSWORD,
  database: process.env.POSTGRES_DB,
});

export async function logPrediction({ operationType, district, surface, propertyType, predictedPriceUsd, modelVersion, latencyMs }) {
  await pool.query(
    `INSERT INTO prediction_logs
       (operation_type, district, surface, property_type, predicted_price_usd, model_version, latency_ms)
     VALUES ($1, $2, $3, $4, $5, $6, $7)`,
    [operationType, district, surface, propertyType, predictedPriceUsd, modelVersion, latencyMs]
  );
}
