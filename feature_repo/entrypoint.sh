#!/bin/sh
# `feast apply` first: even an empty registry (no feature views yet — see
# task 5) needs to exist before `feast serve` can start.
set -e
feast -c /app/feature_repo apply
exec feast -c /app/feature_repo serve --host 0.0.0.0 --port 6566
