#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -m pytest -q
python -m tools.verify_fixture --dir synthetic_fixtures
python -m fededge_experiments.experiments \
  --cohort synthetic_fixtures/synthetic_tiny_smoke_test.npz \
  --outdir artifacts_execution_demo/fresh_run \
  --models full,no_quality_time,late \
  --seeds 7 --method central --epochs 1 --batch 16 --dim 8 --device cpu
printf '\nSYNTHETIC ONLY: performance here is not a clinical or main-paper result.\n'
