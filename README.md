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
