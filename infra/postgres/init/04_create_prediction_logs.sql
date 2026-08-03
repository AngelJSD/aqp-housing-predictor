-- Traceability for the inference API (task 7): one row per /predict
-- request. Chosen over stdout logs because task 9's dashboard needs to
-- query prediction history, not just tail logs.
CREATE TABLE prediction_logs (
    id                  SERIAL PRIMARY KEY,
    requested_at        TIMESTAMP NOT NULL DEFAULT now(),
    operation_type      TEXT NOT NULL,
    district            TEXT NOT NULL,
    surface             DOUBLE PRECISION NOT NULL,
    property_type       TEXT NOT NULL,
    predicted_price_usd DOUBLE PRECISION NOT NULL,
    model_version       TEXT NOT NULL,
    latency_ms          DOUBLE PRECISION NOT NULL
);
