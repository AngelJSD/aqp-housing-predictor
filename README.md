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

Corre el pipeline completo (filtrar Arequipa → deduplicar → completar superficie → descartar incompletos → normalizar moneda → filtrar outliers → imputar geo → guardar) e imprime el conteo de filas en cada paso. Genera `data/processed/listings.parquet` (gitignored, se regenera con este comando).

Cada regla vive como función independiente e importable en `ml/data_prep/clean_arequipa.py` — se reusan tal cual en el paso 10 del plan (lote de anuncios reales actuales), no se duplica la lógica.

Con esto, la tarea 1 del plan está completa. Ver `proyecto-mlops-plan.md` para el detalle de cada decisión.
