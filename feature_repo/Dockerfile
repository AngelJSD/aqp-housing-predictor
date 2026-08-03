# No official feature-server-only image exists for Feast — confirmed
# during task 4 planning. `feast[postgres,redis]` extras verified against
# PyPI metadata for the pinned version.
FROM python:3.12-slim
WORKDIR /app/feature_repo

RUN pip install --no-cache-dir "feast[postgres,redis]==0.65.0"

COPY feature_store.yaml ./
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
