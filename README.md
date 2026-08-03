# Predictor de precio de propiedades — Arequipa

Pipeline MLOps de extremo a extremo: desde el dato crudo hasta un modelo en producción con feature store, tracking, API de inferencia, monitoreo y dashboard.

## Etapa 0 — Setup

Requiere Python 3.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

## Etapa 1 — Datos: descarga y limpieza

Requiere credenciales de la Kaggle API (token en https://www.kaggle.com/settings -> API tokens):

```bash
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
# o: kaggle auth login
# o: kaggle.json legacy en ~/.kaggle/kaggle.json
```

```bash
source .venv/bin/activate
./scripts/download_dataset.sh
```

Descarga `pe_properties.csv` a `data/raw/` (idempotente).

```bash
python3 ml/data_prep/clean_arequipa.py
```

Genera `data/processed/listings.parquet` (gitignored, se regenera con este comando).

## Etapa 2 — Features y catálogo de metadata

```bash
source .venv/bin/activate
python3 ml/data_prep/build_features.py
```

Lee `listings.parquet`, agrega `district_avg_price_per_m2`, guarda `data/processed/features.parquet`.

El catálogo de metadata vive en `ml/feature_metadata.csv` (trackeado en git) hasta cargarse a Postgres en la Etapa 4.

## Etapa 3 — Modelo baseline

Requiere XGBoost (en Mac: `brew install libomp` si falla el import).

```bash
source .venv/bin/activate
python3 ml/training/train.py
```

Entrena y guarda dos modelos XGBoost (Venta/Alquiler) en `data/processed/models/` (gitignored). Imprime R²/RMSE/MAE/MAPE de cada uno.

## Etapa 4 — Infraestructura base (Docker Compose)

Requiere Docker y Docker Compose.

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps   # los 7 servicios deben quedar "healthy"
```

```bash
source .venv/bin/activate
python3 ml/data_prep/load_feature_metadata.py
```

Carga `ml/feature_metadata.csv` a la tabla `feature_metadata` (upsert por `name`, no duplica filas al re-correrlo).

Verificar:

```bash
docker compose exec postgres psql -U aqp -d aqp_housing -c "SELECT * FROM feature_metadata;"   # 5 filas
docker compose exec redis redis-cli ping                                                         # PONG
curl http://localhost:5001/                                                                       # UI de MLflow, HTTP 200
curl http://localhost:6566/health                                                                 # feature server de Feast, HTTP 200
```

## Etapa 5 — Feature store (Feast)

```bash
source .venv/bin/activate
python3 ml/data_prep/load_features_to_postgres.py
```

Reemplaza por completo la tabla `features` en Postgres con el contenido de `features.parquet` (`TRUNCATE` + insert).

```bash
docker compose exec feast feast -c /app/feature_repo materialize 2019-01-01T00:00:00 2020-04-01T00:00:00
```

Empuja los valores del offline store (Postgres) al online store (Redis).

Verificar:

```bash
docker compose exec redis redis-cli DBSIZE   # 6811

curl -s -X POST http://localhost:6566/get-online-features \
  -H "Content-Type: application/json" \
  -d '{
    "features": ["arequipa_listings_features:district", "arequipa_listings_features:surface"],
    "entities": {"id": ["KJlhh3C7WTxl8gf76ZL82g=="]}
  }'
```

## Etapa 6 — Modelo definitivo (Feast + MLflow + ONNX)

Requiere el stack de la Etapa 4, con datos cargados y materializados (Etapa 5).

```bash
source .venv/bin/activate
python3 ml/training/train.py
```

Lee features desde el offline store de Feast (no directo de `features.parquet` como en la Etapa 3). Trackea cada corrida en MLflow (experiment `arequipa-housing-price`) y registra el modelo en el Model Registry (`arequipa-price-venta`, `arequipa-price-alquiler`). UI en `http://localhost:5001`.

```bash
python3 ml/training/export_onnx.py
```

Convierte ambos modelos a ONNX y valida las predicciones contra el modelo original (tolerancia `1e-3`). Guarda `data/processed/models/{venta,alquiler}_xgb.onnx`.

Verificar:

```bash
curl -s http://localhost:5001/api/2.0/mlflow/registered-models/search   # arequipa-price-venta y arequipa-price-alquiler, status READY
```

## Etapa 7 — API de inferencia (Node.js/Fastify)

Requiere los modelos ya entrenados y exportados (Etapa 6) — `data/processed/models/{venta,alquiler}_xgb.{json,onnx,categories.json}` deben existir antes de levantar este servicio.

```bash
docker compose up -d --build api
```

Probar una predicción:

```bash
curl -s -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"district": "Cayma", "surface": 115, "property_type": "Departamento", "operation_type": "Venta"}'
```

Verificar trazabilidad:

```bash
docker compose exec postgres psql -U aqp -d aqp_housing -c "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 5;"
```

## Etapa 8 — Monitoreo (Prometheus + Evidently)

Requiere la API corriendo (Etapa 7).

```bash
docker compose up -d --build prometheus
```

UI en `http://localhost:9090`.

```bash
curl -s "http://localhost:9090/api/v1/query?query=http_requests_total" | python3 -m json.tool
```

```bash
source .venv/bin/activate
python3 ml/monitoring/drift_report.py
```

Compara features de entrenamiento contra `prediction_logs` (tráfico reciente). Un reporte HTML por `operation_type` en `data/processed/drift_reports/` (gitignored).

## Etapa 9 — Dashboard de observabilidad (Next.js)

Requiere el stack completo arriba, con tráfico y reportes generados (Etapas 7/8).

```bash
docker compose up -d --build dashboard
```

UI en `http://localhost:3002`. Secciones: Predictor, Modelos (MLflow), Monitoreo (Prometheus), Drift (Evidently), Catálogo de features.

Probar el predictor: abrir `http://localhost:3002/predictor` y completar el formulario (ej. distrito `Cayma`, superficie `115`, tipo `Departamento`, operación `Venta`).

## Levantar todo de una vez

```bash
cp .env.example .env
docker compose up -d --build
source .venv/bin/activate
python3 ml/data_prep/load_feature_metadata.py
python3 ml/data_prep/load_features_to_postgres.py
docker compose exec feast feast -c /app/feature_repo materialize 2019-01-01T00:00:00 2020-04-01T00:00:00
python3 ml/training/train.py
python3 ml/training/export_onnx.py
docker compose up -d --build api prometheus dashboard
```

Dashboard: `http://localhost:3002` · MLflow: `http://localhost:5001` · Prometheus: `http://localhost:9090`.
