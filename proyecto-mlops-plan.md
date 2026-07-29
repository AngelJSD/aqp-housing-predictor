# Proyecto Final: Pipeline MLOps de extremo a extremo

## Objetivo

Demostrar en un video de 10 minutos el ciclo completo de MLOps aplicado a un caso real: desde el feature store hasta el monitoreo de drift y reentrenamiento del modelo en producción, usando un stack híbrido que combina Python (donde el ecosistema ML es más maduro) con TypeScript/Node.js/Next.js (donde aprovecho mi experiencia previa como ingeniero de software).

## Caso de uso

**Predictor de precio de propiedades en Arequipa, Perú** (dataset "Property Listings for 5 South American Countries", subset `pe_properties.csv`, ~124 mil registros, ~12,400 de Arequipa). Regresión: dado distrito, superficie, tipo de propiedad y tipo de operación (venta/alquiler), estimar el precio. Es una herramienta que un usuario común y corriente entiende y usaría directamente: "¿el precio que me piden es razonable?".

**Nota importante sobre el dataset:** los datos son de 2020. En vez de ser una limitación, esto se aprovecha como ventaja narrativa: en lugar de *simular* drift artificialmente, se compara el modelo entrenado con datos de 2020 contra un pequeño conjunto de anuncios reales y actuales de Arequipa (30-50 propiedades recolectadas a mano de un portal inmobiliario). El drift que aparece es real (inflación, cambios en el mercado post-pandemia, tipo de cambio), no inventado — un argumento más honesto y convincente para el jurado.

**Alcance de la limpieza de datos (deliberadamente mínimo, ya que el foco de evaluación es el pipeline, no el modelo):**
- Deduplicar registros.
- Descartar filas sin precio, superficie o distrito (no imputables sin inventar datos).
- Normalizar moneda a una sola unidad (tipo de cambio fijo razonable para 2020).
- Filtrar outliers de precio por percentiles (ej. descartar 1% superior/inferior).
- Imputar el ~5% de lat/long faltante con el centroide promedio del distrito (o prescindir de esta feature si el distrito ya está limpio).
- Modelo: XGBoost o LightGBM con hiperparámetros por defecto, sin tuning extenso — suficiente para la demo.

## Decisión de arquitectura: por qué Python + Node

En vez de forzar todo el pipeline a un solo lenguaje, se divide el trabajo según dónde cada tecnología aporta más:

- **Python:** entrenamiento del modelo, definición y materialización de features en el feature store, generación de reportes de drift. Son tareas donde el ecosistema (scikit-learn, XGBoost, Feast SDK, Evidently) no tiene equivalente maduro en Node.
- **TypeScript / Node.js / Next.js:** la API de inferencia (sirviendo el modelo exportado a ONNX), el consumo del feature store en tiempo real, y el dashboard de monitoreo. Son las piezas con las que interactúa el usuario final y el jurado, y donde mi experiencia previa permite un resultado más pulido.

Este split es en sí mismo parte del argumento técnico del proyecto: no es "usé lo que ya sabía", sino una decisión de arquitectura defendible sobre dónde vive cada responsabilidad del sistema.

## Stack técnico

| Componente | Tecnología |
|---|---|
| Feature store (offline) | Feast + Postgres |
| Feature store (online) | Feast + Redis |
| Catálogo de metadata de features | Tabla `feature_metadata` en Postgres |
| Entrenamiento y tracking | Python + XGBoost/LightGBM + MLflow |
| Registro de modelos | MLflow Model Registry |
| Formato de despliegue del modelo | ONNX |
| API de inferencia | Node.js/Fastify + onnxruntime-node |
| Métricas de servicio | Prometheus (`prom-client` en Node) |
| Detección de drift | Evidently (Python) |
| Dashboard de monitoreo | Next.js + Recharts |
| Orquestación/infraestructura | Docker + Docker Compose (servidor propio) |

## Estructura del repositorio

Monorepo en GitHub — no se justifican repos separados para un proyecto de este tamaño y plazo. Notebooks solo para EDA exploratorio; todo lo que deba correr de forma repetible dentro del pipeline va como script `.py` normal (un notebook no se ejecuta bien desde Docker/CI).

```
mlops-arequipa-housing/
├── README.md
├── docker-compose.yml
├── .gitignore
├── .env.example
├── data/
│   ├── raw/              # gitignored — el CSV de 107MB NO va al repo
│   │   └── README.md     # instrucciones de dónde descargarlo (link a Kaggle)
│   └── processed/        # gitignored, se regenera con los scripts
├── notebooks/
│   └── 01_eda_arequipa.ipynb
├── ml/                              # todo el mundo Python
│   ├── requirements.txt
│   ├── data_prep/clean_arequipa.py
│   ├── training/train.py
│   └── monitoring/drift_report.py
├── feature_repo/                    # carpeta estándar que espera Feast
│   ├── feature_store.yaml
│   └── features.py
├── api/                             # servicio Node/Fastify (serving)
│   ├── package.json
│   ├── src/
│   └── Dockerfile
├── dashboard/                       # Next.js
│   ├── package.json
│   ├── app/
│   └── Dockerfile
└── docs/
    └── proyecto-mlops-plan.md       # este documento
```

**`.gitignore` clave:** nunca subir el CSV de 107MB (Git no está pensado para eso — solo un `data/raw/README.md` con el link de descarga), además de `node_modules/`, `__pycache__/`, `venv/`/`.venv/`, `.env`, y `mlruns/` si los artifacts de MLflow no se centralizan en el servidor.

**Para la narrativa del video:** los mensajes de commit pueden seguir la misma numeración de esta secuencia de tareas (ej. `feat: paso 1 - limpieza y validación de datos Arequipa`), dando al jurado una traza clara del proceso — reforzando el mismo concepto de trazabilidad que se demuestra en el sistema.

## Secuencia de tareas de alto nivel

1. **Validar y limpiar el raw data de Arequipa.** Filtrar el CSV completo de Perú al subset de Arequipa (~10% del total) y verificar su calidad de forma independiente — el 5% de lat/long faltante identificado antes corresponde al dataset completo de Perú, no necesariamente al subset de Arequipa, así que hay que revisar completitud y nulos específicamente sobre este subset. Hacer la limpieza mínima necesaria (deduplicar, descartar filas sin precio/superficie/distrito, normalizar moneda, filtrar outliers). Este raw data limpio es el equivalente a la tabla operacional de anuncios (`listings`), separada conceptualmente de la tabla de features que viene en el siguiente paso.

   **TODO:**
   - [x] Script reproducible de descarga del dataset (`scripts/download_dataset.sh`, vía Kaggle API).
   - [x] Cargar `pe_properties.csv` completo y filtrar al subset de Arequipa (confirmar columna de distrito/ciudad/departamento a usar como filtro). → `l2 == "Arequipa"` (departamento, 25 valores únicos, 0 nulos), 12,164 filas. `l4` = distrito.
   - [x] EDA de completitud/nulos específico del subset de Arequipa (precio, superficie, distrito, lat/long, tipo de propiedad, tipo de operación) — no asumir el 5% del dataset completo de Perú. → `notebooks/01_eda_arequipa.ipynb`. Hallazgo importante: `surface_total` nulo en 38.1% de filas, `surface_covered` en 57.0% (vs. price 2.2%, currency 2.4%, lat/lon 3.8%, l4/distrito 11.0%).
   - [x] **Decisión: definición de "superficie".** Coalescer `surface_total` con fallback a `surface_covered`, luego descartar filas donde ambas sean nulas. Razonamiento (detalle con números en `notebooks/01_eda_arequipa.ipynb`, sección "Decision: how to define superficie"): superficie tiene correlación log-log moderada con precio (~0.39–0.43, similar en ambas columnas) — distrito y tipo de propiedad probablemente explican más varianza, así que no es una feature crítica al punto de justificar perder >1/3 de los datos. Cuando ambas columnas están presentes, la diferencia mediana es 0 — `surface_covered` es un sustituto legítimo, no una cantidad distinta. Coalescer es un superconjunto estricto de exigir solo `surface_total` (mismas filas + ~800–1,200 más), sin downside. Incluso el peor caso deja miles de filas, más que suficiente para el baseline con hiperparámetros por defecto que pide el plan.
   - [x] Deduplicar registros. → `ml/data_prep/clean_arequipa.py`. Dedup por contenido (todas las columnas salvo `id` y fechas de publicación) — el `id` ya es único por sí solo (0 duplicados exactos), pero 743 filas eran el mismo anuncio republicado bajo otro `id`/fecha. Se conserva la ocurrencia más antigua (`created_on`). 12,164 → 11,421 filas.
   - [x] Descartar filas sin precio, superficie (según la definición coalescida arriba) o distrito. → `coalesce_surface` + `drop_incomplete` en `ml/data_prep/clean_arequipa.py`. 11,421 → 6,975 filas (4,446 descartadas).
   - [x] Normalizar moneda a una sola unidad (definir tipo de cambio fijo razonable para 2020). → `normalize_currency` en `ml/data_prep/clean_arequipa.py`. Todo a USD (moneda mayoritaria en los datos y convención del sector inmobiliario en Perú). Tipo de cambio fijo: S/3.5 por USD (promedio anual BCRP 2020 real: S/3.494). Filas con moneda desconocida (14 de 6,975) se descartan — no convertibles. 6,975 → 6,961 filas.
   - [x] Filtrar outliers de precio por percentiles (ej. 1% superior/inferior). → `filter_price_outliers` en `ml/data_prep/clean_arequipa.py`. Percentiles calculados **por `operation_type`**, no globalmente: el precio de venta mediano es ~150x el de alquiler, así que un corte global apenas tocaría la cola de Venta y cortaría de forma incorrecta alquileres baratos pero legítimos (verificado: 1% global = $243, muy por debajo del 1% real de Venta que es $21,945). 6,961 → 6,825 filas (136 descartadas).
   - [x] Imputar el lat/long faltante con centroide promedio del distrito (o prescindir de la feature si el subset ya viene limpio en ese campo). → `impute_geo` en `ml/data_prep/clean_arequipa.py`. En la práctica es un no-op sobre este dataset: el ~3.8% de nulos crudos de lat/lon se superpone por completo con filas ya descartadas en pasos previos (0 de 6,825 filas llegan sin lat/lon). Se implementó igual como transformación real (no solo un assert) porque se reusa para el lote manual del paso 10, que es otra fuente y podría no tener esa misma superposición.
   - [x] Guardar el resultado como tabla `listings` limpia en `data/processed/`. → `save_listings` en `ml/data_prep/clean_arequipa.py`, formato Parquet (no CSV — preserva dtypes, lo que ya nos mordió una vez con nulos/strings ambiguos en el CSV crudo). `data/processed/listings.parquet`, 6,825 filas × 27 columnas (originales + `surface` y `price_usd` derivadas). Nota de arquitectura: esta tabla NO es todavía el input de entrenamiento — el paso 2 define features sobre ella, el baseline (paso 3) lee esas features directamente sin Feast, y solo el modelo definitivo (paso 6) lee del feature store real de Feast (paso 5).
   - [x] Documentar decisiones de limpieza (conteos antes/después, filas descartadas por motivo) para trazabilidad. → Conteos antes/después impresos en cada paso de `main()` en `clean_arequipa.py`; razonamiento completo de cada decisión en `notebooks/01_eda_arequipa.ipynb` (con código real que llama las funciones del script, sin duplicar lógica) y resumido en este checklist.
   - [x] Encapsular todo en `ml/data_prep/clean_arequipa.py` como script repetible. → Hecho: `load_raw`, `filter_arequipa`, `deduplicate`, `coalesce_surface`, `drop_incomplete`, `normalize_currency`, `filter_price_outliers`, `impute_geo`, `save_listings`, orquestadas en `main()`.

   **Decisión de diseño: `clean_arequipa.py` como funciones reutilizables, no script monolítico.** Cada regla de limpieza (dedup, drop-nulls, coalescer superficie, normalizar moneda, filtrar outliers) va como función importable, con un bloque `if __name__ == "__main__"` delgado que las aplica sobre el CSV raw. Razón: el paso 10 recolecta un lote nuevo de anuncios reales (2020 vs. hoy) que debe pasar por las *mismas* reglas de transformación antes de poder compararse contra el training set o servirse al modelo — si esa lógica se duplica en vez de reusarse, se arriesga skew entre entrenamiento y "producción" (la comparación de drift dejaría de ser honesta). Las mismas funciones se reusan en el paso 10.

2. **Definir las features y su catálogo de metadata.** Con un EDA rápido, decidir qué columnas del raw data son predictivas para el precio (distrito, superficie, tipo de propiedad/operación) y qué derivadas construir (ej. precio histórico promedio por m² por distrito). Documentar cada feature en una tabla `feature_metadata` en Postgres (nombre, descripción, tipo de dato, columna(s) de origen, transformación aplicada, feature view de Feast asociada, owner, fecha de creación, versión) — este catálogo es independiente del registry interno de Feast y sirve como documentación humano-legible que el dashboard puede consultar directamente.

   **Output de esta tarea (dos artefactos distintos, no uno):**
   1. `data/processed/features.parquet` — los *valores* de las features (listings + columnas derivadas, ej. precio promedio por m² por distrito). Esto es lo que lee el baseline (paso 3) para entrenar.
   2. Catálogo `feature_metadata` — *documentación sobre* esas features (nombre, descripción, columna de origen, transformación, owner, versión...), no valores. Empieza como archivo (ver decisión de arquitectura abajo), es un artefacto separado del #1.

   **TODO:**
   - [x] Notebook `notebooks/02_feature_eda.ipynb` (mismo patrón que el paso 1: exploración autocontenida, con código real que importa y llama funciones de un script, no solo prosa) — EDA rápido sobre `data/processed/listings.parquet` para decidir features.
   - [x] Confirmar features base predictivas: distrito (`l4`), superficie (`surface`), tipo de propiedad (`property_type`), tipo de operación (`operation_type`) — validar relación con `price_usd`. → Confirmado en `notebooks/02_feature_eda.ipynb`: distrito abarca ~$75–$1,133/m² dentro de Venta (15x), tipo de propiedad ~$537 (`Lote`) a ~$1,661/m² (`Oficina`). `operation_type` tiene una brecha de escala enorme (precio de venta mediano ~150x el de alquiler) — cualquier estadística derivada debe calcularse **dentro** de cada `operation_type`, igual que la decisión de outliers del paso 1.
   - [x] **Decisión: features derivadas y manejo de leakage.** Precio histórico promedio por m² por distrito, calculado con **leave-one-out suavizado** (smoothing k=10 hacia la media global del `operation_type`), no promedio simple ni LOO puro. Evidencia real en `notebooks/02_feature_eda.ipynb`: el distrito Mollebaya (Venta) tiene solo 6 anuncios, uno con un outlier de $12,500/m² — el promedio simple de ese anuncio queda 203% inflado por su propio valor (leakage real, no teórico). Mollendo tiene solo 1 anuncio — LOO puro da `NaN` ahí. El suavizado resuelve ambos problemas con una sola fórmula: grupos grandes casi no se mueven de su media LOO, grupos pequeños/únicos se contraen hacia la media global en vez de quedar en `NaN` o dominados por un outlier.
   - [x] **Decisión de arquitectura:** el plan pide la tabla `feature_metadata` "en Postgres", pero Postgres recién se levanta en el paso 4. Mientras tanto vive como CSV (`ml/feature_metadata.csv`) — mapea 1:1 a las columnas de la futura tabla, fácil de cargar con `to_sql`/`COPY` en el paso 4. A diferencia de `data/processed/`, este archivo **sí se trackea en git** — es documentación autorada (owner, versión), no dato derivado regenerable.
   - [x] Definir el esquema de `feature_metadata`: `name, description, dtype, source_columns, transformation, feast_feature_view, owner, created_at, version`.
   - [x] Poblar el catálogo con cada feature decidida. → 5 filas en `ml/feature_metadata.csv`: `district`, `surface`, `property_type`, `operation_type`, `district_avg_price_per_m2`. Todas apuntan a la feature view planeada `arequipa_listings_features` (se crea de verdad recién en el paso 5).
   - [x] Encapsular la construcción de features en un script repetible (ej. `ml/data_prep/build_features.py`), mismo patrón de funciones reutilizables que `clean_arequipa.py`. → Hecho: `load_listings`, `smoothed_district_avg`, `build_features`, `save_features`. Lee `data/processed/listings.parquet` (6,825 filas), escribe `data/processed/features.parquet` (6,825 filas × 7 columnas: `id`, 4 features base, 1 derivada, `price_usd` como target). 0 nulos en todas las columnas.

3. **Entrenar un modelo baseline (sin infraestructura todavía).** Entrenar un modelo rápido (XGBoost/LightGBM con default params) directamente sobre la tabla de features ya definida, sin Feast ni Docker todavía. El objetivo es confirmar que el problema es viable (el modelo aprende algo razonable) antes de invertir tiempo en infraestructura.

   **Caveat conocido, no bloqueante para un baseline:** `district_avg_price_per_m2` se calculó como una sola tabla estática sobre las 6,825 filas juntas (leave-one-out solo excluye la propia fila, no separa train/test). Una fila de test todavía puede reflejar en algo el precio de otras filas de test de su mismo distrito. Aceptable para un baseline exploratorio; si esto importa para el modelo definitivo (paso 6), ahí se recalcula solo con el fold de train.

   **TODO:**
   - [x] Elegir librería (XGBoost o LightGBM) e instalar. → **XGBoost**, decidido sin necesidad de comparación empírica (ambas darían calidad similar en un dataset de 6,825 filas). Dos motivos concretos: export a ONNX más maduro (necesario en el paso 6 para la API de inferencia en Node.js), y soporte nativo de categóricas (`enable_categorical=True`), que alimenta directo la siguiente decisión de encoding. Requirió `brew install libomp` (dependencia de OpenMP en Mac). Documentado en `notebooks/03_baseline_model.ipynb`.
   - [x] Decidir encoding de categóricas (`district`, `property_type`, `operation_type`) — soporte nativo de categóricas vs one-hot/label encoding. → **Categóricas nativas** (`enable_categorical=True`), no one-hot. Comparación real en `notebooks/03_baseline_model.ipynb`: nativo gana en R² (0.852 vs 0.819) y MAE, con 5 columnas en vez de 34 (el one-hot de `district`, ~20 valores, no aporta nada aquí). Hallazgo real durante la prueba: el distrito `Camana` quedó solo en el split de test (no visto en train) — caso genuino de categoría no vista, no solo un artefacto del split. Se mapea a `NaN`, que XGBoost trata como missing y rutea nativamente — mismo manejo necesario en `ml/training/train.py` y en inferencia.
   - [x] **Decisión a explorar con evidencia:** ¿un solo modelo con `operation_type` como feature, o dos modelos separados (Venta/Alquiler)? → **Dos modelos separados.** En `notebooks/03_baseline_model.ipynb`: el modelo único entrenado sobre `price_usd` crudo tiene R² = **-2593** en Alquiler (catastrófico, muchísimo peor que predecir la media) — el loss de squared error queda totalmente dominado por la escala de Venta. Entrenando en `log(price_usd)` mejora mucho (R² pasa a -0.08) pero sigue siendo peor que la media. Dos modelos separados (target crudo, antes de decidir log): R²=0.83 en Venta (ya mejor que el unificado en log) y R²=0.40 en Alquiler (vs -0.08) — brecha real que se sostiene sin importar la transformación del target. Además tiene sentido más allá de los números: precio de venta y alquiler mensual son magnitudes económicas distintas (pago único vs. recurrente), no la misma cantidad a otra escala.
   - [x] **Decisión a explorar con evidencia:** ¿entrenar sobre `price_usd` directo o `log(price_usd)`? Decidido por separado para cada modelo (ya que son dos modelos independientes). → **`log(price_usd)` para ambos**, sin necesidad de decisiones distintas — gana claro en R² y MAE en los dos. Venta: R² 0.83→0.87. Alquiler: R² 0.40→**0.62** (el salto más grande). Consistente con la distribución sesgada a la derecha típica de precios inmobiliarios. `ml/training/train.py` entrena ambos modelos en log y aplica `exp()` a las predicciones.
   - [x] Split train/test (definir estrategia: random, estratificado por `operation_type`, etc.). → **Random simple** (`test_size=0.2`, `random_state=42`). Estratificar por `operation_type` ya no aplica (modelos separados). Estratificar por `district` se descartó: `Mollendo` (Venta) y `Camana`/`Characato` (Alquiler) tienen 1 solo anuncio — mecánicamente imposible de estratificar (sklearn exige ≥2 por grupo). El manejo de categorías no vistas (→ `NaN`) ya cubre este caso sin romper nada.
   - [x] Elegir métricas de evaluación (ej. RMSE, MAE, R² — posiblemente en escala log y en escala original). → **R², RMSE, MAE, MAPE**, todas en escala dólar (no log — no interpretable para el jurado). MAPE es la métrica principal (responde directo "¿el precio es razonable?", comparable entre Venta y Alquiler pese a la escala distinta): Venta 15.1%, Alquiler 14.8%. Baseline trivial (predecir la media) da MAPE 119–129% — confirma que el modelo aprende algo real, no solo ruido.
   - [x] Entrenar y registrar métricas del baseline (impresas/loggeadas, no hace falta MLflow todavía — eso es el paso 6). → Tabla final de resultados en `notebooks/03_baseline_model.ipynb`, aplicando todas las decisiones anteriores juntas (encoding nativo, dos modelos, target log, split random, R²/RMSE/MAE/MAPE). Venta: R²=0.8730, MAPE=15.1%. Alquiler: R²=0.6221, MAPE=14.8%.
   - [x] Encapsular en `ml/training/train.py`. → `load_features`, `to_categorical`, `train_operation_model`, `save_model`, orquestadas en `main()`. Guarda cada modelo en `data/processed/models/{venta,alquiler}_xgb.json` (gitignored). El notebook usa el mismo mecanismo de split (`split_operation`: filtrar por `operation_type`, luego split independiente) — las métricas impresas por el script coinciden exactamente con las del notebook, no solo aproximadamente.
   - [x] Confirmar que el modelo aprende algo razonable (mejor que un baseline trivial tipo predecir la media) antes de pasar a infraestructura (paso 4). → Confirmado con margen amplio, no marginal: MAPE ~7.9x menor que el baseline trivial en Venta, ~8.7x menor en Alquiler, R² positivo en ambos. Veredicto documentado en `notebooks/03_baseline_model.ipynb`.

4. **Preparar la infraestructura base.** Levantar los servicios core en Docker Compose: Postgres (incluyendo la tabla `feature_metadata`), Redis, MLflow server, y el Feast feature server.

5. **Construir el feature store.** Definir las feature views en Feast siguiendo exactamente el diseño ya documentado en el catálogo de metadata. Materializar las features al online store.

6. **Entrenar y registrar el modelo definitivo.** Reentrenar (o reusar el baseline validado en el paso 3) leyendo features desde el offline store de Feast, trackear el experimento en MLflow, y registrar la versión ganadora en el Model Registry. Exportar el modelo final a formato ONNX y validar que las predicciones coinciden con el modelo original.

7. **Desplegar la API de inferencia.** Construir el servicio de serving en Node.js: recibe requests, consulta features en tiempo real al Feast feature server, corre la inferencia con el modelo ONNX, y devuelve la predicción. Registrar cada request (input, output, latencia, versión de modelo) para trazabilidad.

8. **Instrumentar monitoreo.** Exponer métricas de la API (latencia, throughput, tasa de error) vía Prometheus. Configurar un job periódico en Python con Evidently que compare la distribución de entrenamiento contra el tráfico reciente y genere reportes de drift.

9. **Construir el dashboard de observabilidad.** Crear un dashboard en Next.js que consuma la API REST de MLflow (versiones y métricas de entrenamiento), las métricas de Prometheus, los reportes de Evidently, y el catálogo `feature_metadata` (sección de documentación de features), presentando todo en un solo lugar.

10. **Demostrar uso real y adaptación del sistema.** Recolectar manualmente un pequeño lote de anuncios inmobiliarios reales y actuales de Arequipa, inyectarlos como "tráfico de producción", generar el reporte de drift correspondiente (drift real 2020 vs. hoy, no simulado), y disparar un reentrenamiento que registre una nueva versión del modelo en MLflow, visible en el dashboard.

11. **Grabar el video final.** Guion de 10 minutos siguiendo el flujo completo: raw data vs. tabla de features → catálogo de metadata → arquitectura y su justificación → feature store → entrenamiento y registro → API de inferencia → un jurado prueba el predictor con una propiedad real de Arequipa desde el dashboard → drift real detectado en vivo (2020 vs. hoy) → reentrenamiento → nueva versión reflejada en el sistema.

## Qué demuestra este proyecto

- Diseño y consumo de un feature store (offline/online, point-in-time correctness).
- Tracking de experimentos y versionado de modelos.
- Despliegue de un modelo en un entorno de producción real (servidor propio, Docker).
- Trazabilidad de las predicciones servidas.
- Monitoreo de infraestructura y de datos/modelo (drift).
- Adaptación del sistema ante cambios en el uso real (reentrenamiento disparado por drift).
- Capacidad de integrar un ecosistema ML basado en Python con un stack de producción en TypeScript/Node.js, tomando decisiones de arquitectura justificadas.

## Mejoras adicionales (si hay tiempo)

Fuera del alcance de las 11 tareas core. Evaluar solo después de tener el pipeline completo funcionando end-to-end — no arriesgar el entregable evaluado por perseguir esto primero.

- **Sistema de ingesta real de anuncios nuevos.** En vez de (o además de) el lote manual único del paso 10, un colector que reutilice las mismas funciones de limpieza de `ml/data_prep/clean_arequipa.py` para: (1) escribir los anuncios nuevos ya limpios en la tabla `listings` de Postgres (fuente de verdad / offline), y (2) correr `feast materialize-incremental` para empujar los valores frescos al online store (Redis) — que es lo que efectivamente consulta la API de inferencia en tiempo real durante la demo. El offline store queda listo para el próximo entrenamiento; el online store es lo que se "siente" en vivo. No requiere un scraper automatizado — puede ser tan simple como un script que recibe un CSV nuevo y corre el mismo pipeline.
