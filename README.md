# Predictor de precio de propiedades — Arequipa

Pipeline MLOps de extremo a extremo. Ver [`proyecto-mlops-plan.md`](./proyecto-mlops-plan.md) para el diseño completo (arquitectura, stack, secuencia de tareas).

Este README documenta cómo correr lo que existe hasta ahora, dividido por etapas. Se irá actualizando a medida que avance el proyecto.

## Etapa 0 — Setup

Requiere Python 3 y la Kaggle API (ver Etapa 1).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

`ml/requirements.txt` cubre todo lo Python del proyecto (descarga de datos, notebooks, y lo que se vaya sumando en pasos futuros: entrenamiento, Feast, MLflow, Evidently).

## Etapa 1 — Datos: descarga y validación del subset de Arequipa

**1.1 Descargar el dataset**

Requiere credenciales de la Kaggle API. Conseguir un token en https://www.kaggle.com/settings -> API tokens, luego cualquiera de:

```bash
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
# o: kaggle auth login
# o: kaggle.json legacy en ~/.kaggle/kaggle.json
```

Con el venv activado:

```bash
./scripts/download_dataset.sh
```

Descarga solo `pe_properties.csv` (no el dataset completo de 5 países) a `data/raw/`. Es idempotente — si el archivo ya existe, no vuelve a descargar. Ver `data/raw/README.md` para detalle.

**1.2 EDA exploratorio (subset de Arequipa)**

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_arequipa.ipynb
```

O interactivo: `jupyter lab notebooks/01_eda_arequipa.ipynb` y correr todas las celdas.

Documenta el razonamiento detrás de cada decisión de limpieza (columna de filtro, definición de superficie, tasa de cambio, outliers por tipo de operación, imputación geo) con datos reales, no solo el resultado.

**1.3 Limpieza → tabla `listings`**

```bash
source .venv/bin/activate
python3 ml/data_prep/clean_arequipa.py
```

Corre el pipeline completo (filtrar Arequipa → deduplicar → completar superficie → descartar incompletos → normalizar moneda → filtrar outliers de precio → filtrar outliers de precio-por-m² → imputar geo → guardar) e imprime el conteo de filas en cada paso. Genera `data/processed/listings.parquet` (gitignored, se regenera con este comando).

Cada regla vive como función independiente e importable en `ml/data_prep/clean_arequipa.py` — se reusan tal cual en el paso 10 del plan (lote de anuncios reales actuales), no se duplica la lógica.

Con esto, la tarea 1 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión.

## Etapa 2 — Features y catálogo de metadata

**2.1 EDA de features**

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_eda.ipynb
```

Confirma que distrito, superficie, tipo de propiedad y tipo de operación son predictivas, y documenta con evidencia real por qué el precio promedio por m² por distrito necesita leave-one-out suavizado (no un promedio simple) para evitar leakage.

**2.2 Construcción de features → tabla `features`**

```bash
source .venv/bin/activate
python3 ml/data_prep/build_features.py
```

Lee `data/processed/listings.parquet`, agrega la feature derivada `district_avg_price_per_m2`, y guarda `data/processed/features.parquet` (gitignored, se regenera con este comando). Esta es la tabla que lee el baseline (paso 3) para entrenar — **no** el feature store real de Feast, que recién se construye en el paso 5.

**2.3 Catálogo `feature_metadata`**

`ml/feature_metadata.csv` — a diferencia de `data/processed/`, este archivo sí se trackea en git (es documentación autorada, no dato regenerable). Vive como CSV porque Postgres todavía no existe (se levanta en el paso 4); se carga a la tabla real ahí.

```bash
python3 -c "import pandas as pd; print(pd.read_csv('ml/feature_metadata.csv'))"
```

Con esto, la tarea 2 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión.

## Etapa 3 — Modelo baseline (sin infraestructura)

**3.1 Exploración de decisiones de modelado**

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/03_baseline_model.ipynb
```

Requiere XGBoost (en Mac: `brew install libomp` si falla el import). Documenta con evidencia real: encoding nativo de categóricas vs one-hot, un modelo unificado vs dos modelos separados (Venta/Alquiler — el unificado da R²=**-567** en Alquiler con target crudo, catastrófico), `price_usd` vs `log(price_usd)` para cada modelo, estrategia de split, y las métricas elegidas (R², RMSE, MAE, MAPE).

También incluye, a partir de una revisión externa: chequeo de overfitting (train vs. test) y una validación Out-of-Time (retener febrero 2020 completo). El OoT reveló un bug de calidad de datos en el paso 1 (filas con `surface` casi 0 y precio/m² absurdo que la limpieza original no filtraba) — **ya corregido** (`filter_surface_sanity` en `clean_arequipa.py`, paso 1).

El overfitting también está resuelto — con un giro interesante: regularización manual a ciegas empeoró el modelo, y tunear sobre un solo split de validación pareció ganar pero colapsó en el split de producción (inestable, no una mejora real). Solo 5-fold cross-validation reveló el problema de verdad (los defaults de Venta tienen R² promedio de **-0.26** entre folds, algo que el split único usado en el resto del notebook ocultaba por completo). `max_depth=5, min_child_weight=3` corrige eso de forma estable para ambos modelos — ya adoptado en `ml/training/train.py`. Detalle completo en `proyecto-mlops-plan.md`, tarea 3.

**3.2 Entrenamiento → modelos baseline**

```bash
source .venv/bin/activate
python3 ml/training/train.py
```

Entrena y guarda dos modelos XGBoost (uno por `operation_type`) en `data/processed/models/` (gitignored). Imprime R²/RMSE/MAE/MAPE de cada uno, comparado contra un baseline trivial (predecir la media) — mismo mecanismo de split que el notebook, mismos números.

Con esto, la tarea 3 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión.

## Etapa 4 — Infraestructura base (Docker Compose)

Requiere Docker y Docker Compose.

**4.1 Variables de entorno**

```bash
cp .env.example .env
```

Un solo Postgres para todo el stack (DB `aqp_housing` para nuestras tablas, DB separada `mlflow` para el backend store de MLflow — ver `proyecto-mlops-plan.md`, tarea 4, para el porqué). Nota Mac: `MLFLOW_HOST_PORT` es 5001, no 5000 — AirPlay Receiver/ControlCenter ya ocupa el 5000.

**4.2 Levantar los 4 servicios**

```bash
docker compose up -d --build
```

Postgres (con la tabla `feature_metadata`, creada por `infra/postgres/init/` al primer arranque), Redis (online store, persistencia AOF), MLflow server (backend en Postgres, artifacts proxied a `./mlruns`), Feast feature server (aplica el registry vacío y sirve — feature views recién se definen en el paso 5).

```bash
docker compose ps   # los 4 deben quedar "healthy"
```

**4.3 Cargar el catálogo `feature_metadata`**

```bash
source .venv/bin/activate
python3 ml/data_prep/load_feature_metadata.py
```

Upsert por `name` (re-correrlo tras editar `ml/feature_metadata.csv` no duplica filas) — separado del init de Postgres a propósito, para no depender de reiniciar el volumen si el CSV cambia.

**4.4 Verificar cada servicio**

```bash
docker compose exec postgres psql -U aqp -d aqp_housing -c "SELECT * FROM feature_metadata;"   # 5 filas
docker compose exec redis redis-cli ping                                                         # PONG
curl http://localhost:5001/                                                                       # UI de MLflow, HTTP 200
curl http://localhost:6566/health                                                                 # feature server de Feast, HTTP 200
```

Con esto, la tarea 4 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión (incluyendo dos bugs reales encontrados y corregidos antes de dar el paso por terminado: el `sslmode` por defecto de Feast contra Postgres, y el manejo de artifacts de MLflow con clientes remotos).

## Etapa 5 — Feature store (Feast)

Requiere el stack de la Etapa 4 arriba (`docker compose up -d`).

**5.1 Cargar los valores de features a Postgres**

```bash
source .venv/bin/activate
python3 ml/data_prep/load_features_to_postgres.py
```

Reemplaza por completo la tabla `features` (Postgres) con el contenido actual de `data/processed/features.parquet` — a diferencia de `feature_metadata` (upsert por nombre), esta tabla se regenera entera cada vez, así que el loader hace `TRUNCATE` + insert masivo. `created_on` (la fecha real del anuncio, ya confirmada idéntica a `start_date` en las 6,811 filas) es el `event_timestamp` que usa Feast para el join point-in-time.

**5.2 Definir las feature views**

`feature_repo/features.py` — `Entity` (`listing`, join key `id`) + `FeatureView` `arequipa_listings_features`, con exactamente las 5 features ya documentadas en `ml/feature_metadata.csv`. Se aplican solas: el `feast apply` del entrypoint (Etapa 4) las recoge en cada `docker compose up`, sin tocar `Dockerfile`/`entrypoint.sh`.

**5.3 Materializar al online store**

```bash
docker compose exec feast feast -c /app/feature_repo materialize 2019-01-01T00:00:00 2020-04-01T00:00:00
```

Empuja los valores del offline store (Postgres) al online store (Redis) — un `materialize` de rango completo, no incremental, porque el dataset es un histórico estático de 2020 (no un stream). Se re-corre a mano cuando cambian los datos, no en cada arranque del contenedor.

**5.4 Verificar**

```bash
docker compose exec redis redis-cli DBSIZE   # 6811 — una key por listing

curl -s -X POST http://localhost:6566/get-online-features \
  -H "Content-Type: application/json" \
  -d '{
    "features": ["arequipa_listings_features:district", "arequipa_listings_features:surface"],
    "entities": {"id": ["KJlhh3C7WTxl8gf76ZL82g=="]}
  }'
```

Con esto, la tarea 5 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión, incluyendo la validación cruzada real (fuente vs. offline store vs. online store, mismo `id`, mismos valores).

## Etapa 6 — Modelo definitivo (Feast + MLflow + ONNX)

Requiere el stack de la Etapa 4 arriba, con datos cargados y materializados (Etapa 5).

**6.1 Entrenar, trackear y registrar**

```bash
source .venv/bin/activate
python3 ml/training/train.py
```

Lee las 4 features base desde el offline store de Feast (`get_historical_features`), en vez de `data/processed/features.parquet` directo como en la Etapa 3. Cambio importante respecto al baseline: `district_avg_price_per_m2` (la 5ª feature del catálogo) se **elimina del modelo** — recalcularla sin leakage (fit solo en el fold de train, como quedó pendiente desde la Etapa 3) resultó *peor* y más inestable que no tenerla en absoluto; el detalle completo de esa investigación (no solo la conclusión) está en `notebooks/04_definitive_model.ipynb`. `MODEL_PARAMS` fue re-tuneado por 5-fold CV para el nuevo set de 3 features.

Cada corrida trackea en MLflow (experiment `arequipa-housing-price`, un run por `operation_type`) y registra el modelo en el Model Registry (`arequipa-price-venta`, `arequipa-price-alquiler`). UI en `http://localhost:5001`.

**6.2 Exportar a ONNX y validar**

```bash
python3 ml/training/export_onnx.py
```

Convierte ambos modelos a ONNX (`onnxmltools`) y valida las predicciones contra el modelo original sobre el test set completo de cada uno, tolerancia explícita (`1e-3`). El modelo con categóricas nativas (`enable_categorical=True`, decisión de la Etapa 3) exporta directo — no hizo falta ningún fallback a one-hot/ordinal, contra lo que se esperaba inicialmente. Guarda `data/processed/models/{venta,alquiler}_xgb.onnx` (gitignored, igual que los `.json`).

**6.3 Verificar**

```bash
curl -s http://localhost:5001/api/2.0/mlflow/registered-models/search   # arequipa-price-venta y arequipa-price-alquiler, status READY
```

Con esto, la tarea 6 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión — en particular, la investigación real detrás de por qué se eliminó `district_avg_price_per_m2` (no fue un ajuste menor: un solo split cayó a R²=-0.61 en Venta con los hiperparámetros del baseline, confirmado no ser un bug antes de decidir el fix) y los dos gotchas reales de la exportación a ONNX (nombres de feature `f%d`, códigos de categoría en vez de strings).

## Etapa 7 — API de inferencia (Node.js/Fastify)

Requiere el stack completo arriba, con los modelos ya entrenados y exportados (Etapa 6) — `data/processed/models/{venta,alquiler}_xgb.{json,onnx,categories.json}` tienen que existir antes de levantar este servicio.

**Decisión importante, se aparta de la redacción original del plan:** el endpoint `/predict` **no consulta a Feast en tiempo real**. La única feature que lo hubiera justificado (`district_avg_price_per_m2`, para una propiedad nueva sin `id` en el feature store) se eliminó del modelo en la Etapa 6 — no aportaba nada real más allá de `district`. Mostrarla igual como "contexto" en la UI se descartó a propósito: es el mismo precio por m² ya reflejado en la predicción, presentado como si fuera una señal independiente — confunde más de lo que aclara. El online store de Feast ya quedó demostrado de punta a punta en la Etapa 5; no hacía falta repetirlo acá sin una necesidad real del producto. Detalle completo en `proyecto-mlops-plan.md`, tarea 7.

**7.1 Levantar el servicio**

```bash
docker compose up -d --build api
```

Carga los dos modelos ONNX al arrancar y resuelve la versión registrada de cada uno contra el Model Registry de MLflow (cacheada, no por request). Nota real: `node:22-alpine` no sirve para este servicio — `onnxruntime-node` necesita glibc, y falla en musl (`ERR_DLOPEN_FAILED`); el `Dockerfile` usa `node:22-slim`.

**7.2 Probar una predicción**

```bash
curl -s -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"district": "Cayma", "surface": 115, "property_type": "Departamento", "operation_type": "Venta"}'
```

Un distrito no visto en entrenamiento (ej. uno inventado) no da error — se enruta como *missing*, igual que en Python (`onnxruntime-node` probado directo contra `onnxruntime` de Python con el mismo caso, coinciden).

**7.3 Verificar trazabilidad**

```bash
docker compose exec postgres psql -U aqp -d aqp_housing -c "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 5;"
```

Cada predicción exitosa queda registrada (input, output, latencia, versión de modelo) — un 400 de validación (falta un campo, `operation_type` inválido) correctamente no genera una fila.

Con esto, la tarea 7 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión, incluyendo el bug real de `--allowed-hosts` de MLflow que solo se manifestó al conectar desde otro contenedor (mismo mecanismo de seguridad que ya había aparecido en la Etapa 4, pero recién disparado ahora).

## Etapa 8 — Monitoreo (Prometheus + Evidently)

Requiere el stack completo arriba, con la API corriendo (Etapa 7).

**8.1 Levantar Prometheus**

```bash
docker compose up -d --build prometheus
```

Scrapea `api:3000/metrics` cada 15s (`infra/prometheus/prometheus.yml`). UI en `http://localhost:9090`.

```bash
curl -s "http://localhost:9090/api/v1/query?query=http_requests_total" | python3 -m json.tool
```

Latencia (`http_request_duration_seconds`) y throughput (`http_requests_total`), ambos labeleados por `method`/`route`/`status_code` — la tasa de error se deriva del label `status_code`, no hace falta un contador aparte.

**8.2 Generar reportes de drift**

```bash
source .venv/bin/activate
python3 ml/monitoring/drift_report.py
```

Compara `district`/`surface`/`property_type` de la tabla `features` (entrenamiento) contra `prediction_logs` (tráfico reciente vía la API) — drift de **features de entrada**, no de exactitud de predicción (no hay precio real para el tráfico servido). Un reporte HTML por `operation_type` en `data/processed/drift_reports/` (gitignored). Con los pocos requests de prueba disponibles hoy (misma distribución de 2020) no hay drift interesante que mostrar todavía — eso es explícitamente el paso 10 (lote real de anuncios actuales).

Con esto, la tarea 8 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión, incluyendo un bug real de entorno (un hook de seguridad de NLTK, dependencia transitiva de Evidently, bloqueaba el import porque el `.venv` de este proyecto vive dentro del repo — falso positivo confirmado leyendo su código fuente, resuelto con la variable de escape que la propia librería expone).

## Etapa 9 — Dashboard de observabilidad (Next.js)

Requiere el stack completo arriba, con tráfico y reportes generados (Etapas 7/8) para que haya algo real que mostrar.

**Decisión de arquitectura:** todas las llamadas a servicios externos (MLflow, Prometheus, Postgres, la API de inferencia) pasan por el backend de Next.js — el navegador solo habla con el propio dashboard, nunca directo con otro servicio. Evita configurar CORS en la API y el gotcha ya conocido de este proyecto (código server-side necesita nombres de servicio internos, código en el navegador necesita puertos mapeados al host). El formulario de predicción interactivo (no estaba en la lista original del plan para este paso, pero el guion del video sí lo exige) también pasa por un proxy server-side. Detalle completo en `proyecto-mlops-plan.md`, tarea 9.

**9.1 Levantar el dashboard**

```bash
docker compose up -d --build dashboard
```

UI en `http://localhost:3002`. Secciones: Predictor, Modelos (MLflow), Monitoreo (Prometheus), Drift (Evidently), Catálogo de features (`feature_metadata`).

**9.2 Probar el predictor**

Abrir `http://localhost:3002/predictor` y completar el formulario (ej. distrito `Cayma`, superficie `115`, tipo `Departamento`, operación `Venta`) — llama a la API de inferencia server-side y muestra la estimación junto con una nota de confianza (Venta tiene más variabilidad que Alquiler entre folds de CV, ver paso 3).

Con esto, la tarea 9 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión, incluyendo dos bugs reales de red en el contenedor del dashboard: Docker fija `HOSTNAME` al ID del contenedor (el servidor standalone de Next hace `HOSTNAME || '0.0.0.0'`, así que sin corregirlo termina escuchando solo en la IP interna, no en `0.0.0.0`) y `wget` en Alpine resuelve `localhost` a IPv6 antes que IPv4 (el healthcheck usa `127.0.0.1` explícito por eso). También un bug real detectado con `next build` (no solo `next dev`): la página del catálogo de features se prerenderizaba en build time con datos de la DB horneados de forma estática — corregido con `force-dynamic`.
