import { Pool } from "pg";

const pool = new Pool({
  host: process.env.POSTGRES_HOST,
  port: Number(process.env.POSTGRES_PORT),
  user: process.env.POSTGRES_USER,
  password: process.env.POSTGRES_PASSWORD,
  database: process.env.POSTGRES_DB,
});

export type FeatureMetadata = {
  name: string;
  description: string;
  dtype: string;
  source_columns: string;
  transformation: string;
  feast_feature_view: string;
  owner: string;
  created_at: string;
  version: number;
};

export async function getFeatureMetadata(): Promise<FeatureMetadata[]> {
  const { rows } = await pool.query("SELECT * FROM feature_metadata ORDER BY name");
  return rows;
}

export type PredictionLog = {
  id: number;
  requested_at: string;
  operation_type: string;
  district: string;
  surface: number;
  property_type: string;
  predicted_price_usd: number;
  model_version: string;
  latency_ms: number;
};

export async function getRecentPredictions(limit = 20): Promise<PredictionLog[]> {
  const { rows } = await pool.query("SELECT * FROM prediction_logs ORDER BY requested_at DESC LIMIT $1", [limit]);
  return rows;
}
