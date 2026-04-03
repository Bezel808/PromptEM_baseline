#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_TAG="${SPLIT_TAG:-040303}"
DATA_ROOT="${DATA_ROOT:-/home/zongze/mengshichen_projects/datasets_joint_discovery_integration_split_work}"
DATASETS="${DATASETS:-wikidbs santos_benchmark magellan}"
SEEDS="${SEEDS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUNS_BASE="${RUNS_BASE:-/home/zongze/mengshichen_projects/runs/baseline_smoke_${SPLIT_TAG}}"
LOGS_BASE="${LOGS_BASE:-/home/zongze/mengshichen_projects/logs/baseline_smoke_${SPLIT_TAG}}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
STRICT_EVAL="${STRICT_EVAL:-1}"
PROFILE="${PROFILE:-0}"
PROFILE_INTERVAL_SEC="${PROFILE_INTERVAL_SEC:-1.0}"
PROFILE_OUT="${PROFILE_OUT:-}"

if [[ -n "${PROMPTEM_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PROMPTEM_PYTHON_BIN}"
elif [[ -x "/home/zongze/.venvs/promptem/bin/python" ]]; then
  PYTHON_BIN="/home/zongze/.venvs/promptem/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PromptEM python not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import pandas        # noqa: F401
import numpy         # noqa: F401
import torch         # noqa: F401
import transformers  # noqa: F401
print("[PRECHECK] promptem python imports ok")
PY

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[PRECHECK] promptem wrapper ok"
  exit 0
fi

mkdir -p "${RUNS_BASE}/promptem" "${LOGS_BASE}/promptem"

overall_rc=0
for seed in ${SEEDS}; do
  seed_tag="seed${seed}"
  run_root="${RUNS_BASE}/promptem/${seed_tag}"
  log_root="${LOGS_BASE}/promptem/${seed_tag}"
  mkdir -p "${run_root}" "${log_root}"

  ts="$(date +%Y%m%d_%H%M%S)"
  run_log="${log_root}/run_0323_promptem_${SPLIT_TAG}_${seed_tag}_${ts}.log"

  cmd=(
    env
    "PYTHONHASHSEED=${seed}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "SEED=${seed}"
    "MODEL_NAME_OR_PATH=roberta-base"
    "K=0.1"
    "BATCH_SIZE=32"
    "LR=2e-5"
    "TEACHER_EPOCHS=20"
    "STUDENT_EPOCHS=30"
    "DYNAMIC_DATASET=8"
    "TRANSFORMERS_OFFLINE=0"
    "HF_HUB_OFFLINE=0"
    "HF_DATASETS_OFFLINE=0"
    "DATA_ROOT_BASE=${DATA_ROOT}"
    "SPLIT_TAG=${SPLIT_TAG}"
    "DATASETS=${DATASETS}"
    "RUNS_ROOT=${run_root}"
    "STRICT_EVAL=${STRICT_EVAL}"
    "PROFILE=${PROFILE}"
    "PROFILE_INTERVAL_SEC=${PROFILE_INTERVAL_SEC}"
    "PROFILE_OUT=${PROFILE_OUT}"
    bash "${SCRIPT_DIR}/run_promptem_split.sh"
  )

  echo "[PROMPTEM][${seed_tag}] log=${run_log}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] %q ' "${cmd[@]}"
    echo
    continue
  fi

  "${cmd[@]}" 2>&1 | tee "${run_log}"
  rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    overall_rc=${rc}
    echo "[PROMPTEM][${seed_tag}] failed rc=${rc}" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit ${rc}
    fi
  else
    echo "[PROMPTEM][${seed_tag}] done"
  fi
done

exit ${overall_rc}
