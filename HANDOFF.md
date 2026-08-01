# Handoff — starting the infra stage (task 4)

Context for a fresh Claude instance picking up this project. Read `proyecto-mlops-plan.md` in full before doing anything — it's the source of truth (scope, every decision made with its reasoning, TODO checklists per task). `README.md` has copy-pasteable run commands for everything done so far.

## State: tasks 1–3 done, task 4 next

- **Task 1** (raw CSV → clean `listings` table) — done.
- **Task 2** (features + `feature_metadata` catalog) — done.
- **Task 3** (baseline model) — done, including follow-up hardening: overfitting was found (train/test gap), investigated, and fixed via CV-validated regularization; an Out-of-Time validation was added; a real data-quality bug (13 rows with near-zero `surface`) was found by that OoT check and fixed at the source (`clean_arequipa.py`). Two trained XGBoost models exist: `data/processed/models/{venta,alquiler}_xgb.json`.
- **Task 4** (Docker Compose infra: Postgres w/ `feature_metadata` table, Redis, MLflow server, Feast feature server) — **not started. This is the next job.**

## Workflow conventions established in tasks 1–3 — keep following these

1. Each plan task gets a **TODO checklist** added under it in `proyecto-mlops-plan.md` before starting real work — propose it, let the user review, then execute item by item.
2. Real open decisions get explored with **actual computed evidence** (run the numbers, don't assume) before deciding, and get documented with that evidence, not just the conclusion.
3. Each TODO item, once done, gets checked `[x]` with a `→` note summarizing what was built + real numbers.
4. Scripts (`ml/data_prep/clean_arequipa.py`, `build_features.py`, `ml/training/train.py`) follow a "small reusable functions + thin `main()`" pattern — each transform is its own importable function.
5. Notebooks are EDA/decision-documentation only — self-contained (no "per the plan" references; those get reworded so the notebook stands alone), and where a script already exists, the notebook **imports and calls its real functions** instead of re-deriving logic, to avoid drift between the two.
6. After any notebook edit: re-execute with `jupyter nbconvert --to notebook --execute --inplace <path>` and check the actual output before calling it done.
7. `README.md` gets a new "Etapa N" section per task with runnable commands.
8. If work incidentally surfaces a bug/issue outside current scope, flag it and ask before fixing — don't silently expand scope. (Happened once: an OoT check surfaced a task-1 data-quality bug; it got flagged, confirmed with the user, then fixed.)

## Setup

```bash
source .venv/bin/activate   # venv already exists at repo root
pip install -r ml/requirements.txt
```

XGBoost needs `brew install libomp` on Mac (already installed on this machine).

## Known gotchas

- **`NotebookEdit` tool bug**: it has silently taken a literal `\n` in a prompt as two characters instead of an actual newline, corrupting a cell (renders with visible backslashes, or a Python `SyntaxError: unexpected character after line continuation`). If that happens, rewrite the notebook file directly with `Write`, using real JSON with actual newlines in the `source` arrays — don't fight `NotebookEdit` for it.
- **Split-mechanism mismatches**: comparing two pieces of code that both call `train_test_split` on "the same data" — splitting the full dataset then filtering by a condition vs. filtering first then splitting independently gives *different* splits even with the same `random_state`, because the shuffle depends on array size. Bit us once. If a notebook and a script are meant to reproduce the same numbers, make sure they split identically.
- **Single-split hyperparameter tuning is unreliable** on this dataset's size (~2,000–4,800 rows per model) — a config that looks best on one validation split can be *worse* on a different split of the same data. Use k-fold CV to validate any hyperparameter choice, not a single split. (This is why `train.py`'s current `MODEL_PARAMS` were chosen via CV, not a single validation split — see `notebooks/03_baseline_model.ipynb`, "Regularization" section, for the full story including two dead ends.)
- **Kaggle CLI auth**: `scripts/download_dataset.sh` supports both `KAGGLE_API_TOKEN` (newer) and legacy `kaggle.json`/`kaggle auth login`.

## Task 4 specifics

Plan's exact scope for task 4: "Levantar los servicios core en Docker Compose: Postgres (incluyendo la tabla `feature_metadata`), Redis, MLflow server, y el Feast feature server."

From the plan's "Stack técnico" table:
- Feature store offline: Feast + Postgres
- Feature store online: Feast + Redis
- Feature metadata catalog: Postgres table `feature_metadata` — **a CSV seed is already ready**: `ml/feature_metadata.csv` (5 rows, schema documented in task 2's plan section). Loading it into the real Postgres table is task 4's (or maybe early task 5's) job — the plan doesn't pin down exactly which; use judgment and propose it in the task 4 TODO list.
- Experiment tracking: MLflow server
- Orchestration: Docker + Docker Compose, "servidor propio" (own server, not managed cloud)

Nothing infra-related exists yet: no `docker-compose.yml`, no `.env.example`, no `feature_repo/` directory. The plan's "Estructura del repositorio" section sketches the intended layout — note it shows `proyecto-mlops-plan.md` living under `docs/`, but in practice the file is at repo root; that's an intentional, already-established deviation, don't "fix" it.

**Not task 4's job** (later tasks — don't scope-creep into them): Feast feature *view* definitions (task 5), model registration in MLflow (task 6), ONNX export (task 6), the Node.js inference API (task 7), Prometheus/Evidently (task 8).

## Uncommitted changes right now

`git status` shows modifications (not new files) to `README.md`, `ml/data_prep/clean_arequipa.py`, `ml/training/train.py`, `notebooks/01_eda_arequipa.ipynb`, `notebooks/02_feature_eda.ipynb`, `notebooks/03_baseline_model.ipynb`, `proyecto-mlops-plan.md` — all from this session's tasks 2–3 work, not yet committed. Two prior commits exist on the branch. Don't commit anything without being asked.

## Where everything lives

| File | Purpose |
|---|---|
| `proyecto-mlops-plan.md` | Full plan, decision log, TODO checklists. Read first. |
| `README.md` | Run commands for every stage done so far. |
| `ml/data_prep/clean_arequipa.py` | Raw CSV → `data/processed/listings.parquet`. |
| `ml/data_prep/build_features.py` | Listings → `data/processed/features.parquet` (+ `district_avg_price_per_m2`). |
| `ml/feature_metadata.csv` | Feature catalog — tracked in git (unlike `data/processed/`, which is gitignored/regenerable). |
| `ml/training/train.py` | Trains + saves both XGBoost models to `data/processed/models/`. |
| `notebooks/01_eda_arequipa.ipynb` | Task 1 decision trail (cleaning). |
| `notebooks/02_feature_eda.ipynb` | Task 2 decision trail (features). |
| `notebooks/03_baseline_model.ipynb` | Task 3 decision trail (baseline model, overfitting, OoT, regularization). |

This file (`HANDOFF.md`) is scratch — safe to delete once task 4 is underway and this context is no longer needed.
