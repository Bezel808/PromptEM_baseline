#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

COMMON_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PROFILE_HELPERS="$COMMON_ROOT/baseline_common/profiling_helpers.sh"
AGGREGATE_PY="$COMMON_ROOT/baseline_common/profiling_aggregate.py"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for cand in \
    "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/venv/bin/python" \
    "$HOME/.venvs/promptem/bin/python" \
    "/home/zongze/.venvs/promptem/bin/python"
  do
    if [[ -x "$cand" ]]; then
      PYTHON_BIN="$cand"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] python3 not found. Set PYTHON_BIN=/path/to/python" >&2
    exit 1
  fi
fi

DATA_ROOT_BASE="${DATA_ROOT_BASE:-}"
if [[ -z "$DATA_ROOT_BASE" ]]; then
  for cand in \
    "$REPO_ROOT/../A-joint" \
    "$REPO_ROOT/../../A-joint" \
    "$REPO_ROOT/data_1218" \
    "/home/mengshi/table_quality/datasets_joint_discovery_integration"
  do
    if [[ -d "$cand" ]]; then
      DATA_ROOT_BASE="$cand"
      break
    fi
  done
fi
if [[ -z "${DATA_ROOT_BASE:-}" || ! -d "$DATA_ROOT_BASE" ]]; then
  echo "[ERROR] DATA_ROOT_BASE not found. Set DATA_ROOT_BASE=/path/to/datasets_root" >&2
  exit 1
fi

DATASETS="${DATASETS:-wikidbs_1218 santos_benchmark_1218 magellan_1218}"
GPU="${GPU:-}"

STRICT_EVAL="${STRICT_EVAL:-1}"
PROFILE="${PROFILE:-0}"
PROFILE_INTERVAL_SEC="${PROFILE_INTERVAL_SEC:-1.0}"
PROFILE_OUT="${PROFILE_OUT:-}"

# PromptEM baseline for this paper run should always allow online HF fetch.
unset TRANSFORMERS_OFFLINE || true
unset HF_HUB_OFFLINE || true
unset HF_DATASETS_OFFLINE || true
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

DEVICE="${DEVICE:-cuda}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-roberta-base}"
K="${K:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-2e-5}"
NUM_ITER="${NUM_ITER:-1}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-30}"
TEST_EVERY="${TEST_EVERY:-1}"
SEED="${SEED:-2022}"
ONLY_PLM="${ONLY_PLM:-0}"
SELF_TRAIN="${SELF_TRAIN:-1}"
DYNAMIC_DATASET="${DYNAMIC_DATASET:-8}"
PSEUDO_LABEL_METHOD="${PSEUDO_LABEL_METHOD:-uncertainty}"
UNCERTAINTY_RATIO="${UNCERTAINTY_RATIO:-0.1}"
EL2N_RATIO="${EL2N_RATIO:-0.1}"
MC_DROPOUT_PASS="${MC_DROPOUT_PASS:-10}"
TEMPLATE_NO="${TEMPLATE_NO:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
EXTRA_ARGS=("$@")

MAGELLAN_TEACHER_EPOCHS="${MAGELLAN_TEACHER_EPOCHS:-${TEACHER_EPOCHS}}"
MAGELLAN_STUDENT_EPOCHS="${MAGELLAN_STUDENT_EPOCHS:-30}"
MAGELLAN_TEST_EVERY="${MAGELLAN_TEST_EVERY:-${TEST_EVERY}}"

RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs/promptem_em}"
TS="$(date +%Y%m%d_%H%M%S)"
K_TAG="${K//./p}"
RUN_DIR="${RUNS_ROOT}/k${K_TAG}_${TS}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_MD="${RUN_DIR}/summary.md"

cat > "$SUMMARY_MD" <<'MD'
| Dataset | Precision | Recall | F1 | Accuracy | AUC | Log |
| ------- | --------: | -----: | --:| -------: | --: | --- |
MD

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
Path("$SUMMARY_JSON").write_text(json.dumps([], indent=2), encoding="utf-8")
PY

if [[ -f "$PROFILE_HELPERS" ]]; then
  # shellcheck source=/home/zongze/mengshichen_projects/baseline_common/profiling_helpers.sh
  source "$PROFILE_HELPERS"
else
  echo "[WARN] profiling helpers not found: $PROFILE_HELPERS" >&2
fi

cleanup_profile_sampler_on_exit() {
  if [[ "${PROFILE:-0}" != "1" ]]; then
    return 0
  fi
  if [[ "$(type -t baseline_profile_stop || true)" != "function" ]]; then
    return 0
  fi
  if [[ -n "${BASELINE_PROFILE_SAMPLER_PID:-}" ]]; then
    baseline_profile_stop "" >/dev/null 2>&1 || true
  fi
}
trap cleanup_profile_sampler_on_exit EXIT

if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "========================================="
echo "Start: $(date)"
echo "Repo Root: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Data Root: $DATA_ROOT_BASE"
echo "GPU: ${GPU:-<not set>}"
echo "Device: $DEVICE"
echo "Datasets: $DATASETS"
echo "Strict Eval: $STRICT_EVAL"
echo "Profile: $PROFILE (interval=${PROFILE_INTERVAL_SEC}s)"
echo "Run Dir: $RUN_DIR"
echo "========================================="

run_dataset() {
  local dataset="$1"
  local log_file="$LOG_DIR/${dataset}.log"
  local dataset_root="$DATA_ROOT_BASE/$dataset"
  local promptem_data_dir="$REPO_ROOT/data/$dataset"
  local local_teacher_epochs="$TEACHER_EPOCHS"
  local local_student_epochs="$STUDENT_EPOCHS"
  local local_test_every="$TEST_EVERY"
  local convert_elapsed="0.0"
  local train_elapsed="0.0"
  local profiling_json=""
  local profile_dir="${RUN_DIR}/profiling/${dataset}"

  if [[ "$dataset" == magellan_* ]]; then
    local_teacher_epochs="$MAGELLAN_TEACHER_EPOCHS"
    local_student_epochs="$MAGELLAN_STUDENT_EPOCHS"
    local_test_every="$MAGELLAN_TEST_EVERY"
  fi

  echo "[$(date '+%F %T')] [DATASET=$dataset] START" | tee "$log_file"

  if [[ ! -d "$dataset_root" ]]; then
    echo "[ERROR] dataset root not found: $dataset_root" | tee -a "$log_file"
    return 1
  fi
  for req in \
    "$dataset_root/datalake_plus" \
    "$dataset_root/label_plus/entity_matching/train.csv" \
    "$dataset_root/label_plus/entity_matching/validate.csv" \
    "$dataset_root/label_plus/entity_matching/test.csv"
  do
    if [[ ! -e "$req" ]]; then
      echo "[ERROR] preflight failed, missing: $req" | tee -a "$log_file"
      return 1
    fi
  done

  mkdir -p "$profile_dir"
  if declare -F baseline_profile_start >/dev/null 2>&1; then
    baseline_profile_start "${PROFILE_OUT:-$profile_dir}" "${dataset}_${TS}"
  fi

  if [[ "$FORCE_CONVERT" == "1" || ! -f "$promptem_data_dir/manifest.json" ]]; then
    local t0_convert
    t0_convert="$(date +%s.%N)"
    echo "[$(date '+%F %T')] [DATASET=$dataset] convert 1218 -> PromptEM" | tee -a "$log_file"
    "$PYTHON_BIN" "$REPO_ROOT/convert_1218_to_promptem.py" \
      --dataset-root "$dataset_root" \
      --output-dir "$promptem_data_dir" \
      --max-cell-chars 200 \
      --skip-empty 2>&1 | tee -a "$log_file"
    convert_elapsed="$($PYTHON_BIN - <<PY
from decimal import Decimal
from time import time
print(float(Decimal(str(time())) - Decimal("$t0_convert")))
PY
)"
  else
    echo "[$(date '+%F %T')] [DATASET=$dataset] skip convert (manifest exists)" | tee -a "$log_file"
  fi

  cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/main.py"
    -d "$dataset"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --device "$DEVICE"
    -k "$K"
    -bs "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --lr "$LR"
    -ni "$NUM_ITER"
    -te "$local_teacher_epochs"
    -se "$local_student_epochs"
    --test_every "$local_test_every"
    -pm "$PSEUDO_LABEL_METHOD"
    -ur "$UNCERTAINTY_RATIO"
    -er "$EL2N_RATIO"
    -mdp "$MC_DROPOUT_PASS"
    -tn "$TEMPLATE_NO"
    --seed "$SEED"
  )

  if [[ "$STRICT_EVAL" == "1" ]]; then
    cmd+=(--strict-eval)
  else
    cmd+=(--no-strict-eval)
  fi

  if [[ "$SELF_TRAIN" == "1" ]]; then
    cmd+=(-st)
  fi

  if [[ "$DYNAMIC_DATASET" != "-1" ]]; then
    cmd+=(-dd "$DYNAMIC_DATASET")
  fi

  if [[ "$ONLY_PLM" == "1" ]]; then
    cmd+=(--only_plm)
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  echo "[$(date '+%F %T')] [DATASET=$dataset] run: ${cmd[*]}" | tee -a "$log_file"
  local t0_train
  t0_train="$(date +%s.%N)"
  "${cmd[@]}" 2>&1 | tee -a "$log_file"
  train_elapsed="$($PYTHON_BIN - <<PY
from decimal import Decimal
from time import time
print(float(Decimal(str(time())) - Decimal("$t0_train")))
PY
)"

  if declare -F baseline_profile_stop >/dev/null 2>&1; then
    local stage_times_json
    stage_times_json="$(printf '{"convert": %.6f, "train": %.6f}' "$convert_elapsed" "$train_elapsed")"
    profiling_json="$(baseline_profile_stop "$stage_times_json")"
  fi

  "$PYTHON_BIN" - "$dataset" "$log_file" "$SUMMARY_MD" "$SUMMARY_JSON" "$profiling_json" "$STRICT_EVAL" <<'PY'
import json
import re
import sys
from pathlib import Path

dataset = sys.argv[1]
log_path = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
summary_json = Path(sys.argv[4])
profiling_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
strict_enabled = len(sys.argv) > 6 and sys.argv[6] == "1"

text = log_path.read_text(encoding="utf-8", errors="ignore")

patterns_with_auc = [
    r"\[Best in iter#\d+\] Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+), Accuracy: ([0-9.]+), AUC: ([0-9.]+)",
    r"\[Best Teacher in iter#\d+\] Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+), Accuracy: ([0-9.]+), AUC: ([0-9.]+)",
]
patterns_legacy = [
    r"\[Best in iter#\d+\] Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+)",
    r"\[Best Teacher in iter#\d+\] Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+)",
]
match = None
for p in patterns_with_auc:
    found = re.findall(p, text)
    if found:
        match = found[-1]
        break
if match is not None:
    precision, recall, f1, accuracy, auc = map(float, match)
else:
    for p in patterns_legacy:
        found = re.findall(p, text)
        if found:
            match = found[-1]
            break
    if match is None:
        raise SystemExit(f"Cannot find best metric line in {log_path}")
    precision, recall, f1 = map(float, match)
    accuracy, auc = None, None

strict_eval = None
strict_hits = re.findall(r"\[STRICT_EVAL_JSON\]\s*(\{.*\})", text)
for blob in reversed(strict_hits):
    try:
        strict_eval = json.loads(blob)
        break
    except Exception:
        continue

if strict_eval is None and strict_enabled:
    valid_total = 0
    test_total = 0
    valid_sizes = re.findall(r"valid size: (\d+)", text)
    test_sizes = re.findall(r"test size: (\d+)", text)
    if valid_sizes:
        valid_total = int(valid_sizes[-1])
    if test_sizes:
        test_total = int(test_sizes[-1])
    strict_eval = {
        "enabled": True,
        "threshold_source": "validate",
        "threshold": 0.5,
        "valid_metrics": {},
        "test_metrics": {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": (None if accuracy is None else float(accuracy)),
            "auc": (None if auc is None else float(auc)),
        },
        "coverage": {
            "valid_total": valid_total,
            "valid_used": valid_total,
            "test_total": test_total,
            "test_used": test_total,
        },
        "skipped": {"valid": 0, "test": 0},
        "failed": {"valid": 0, "test": 0},
    }

test_metrics = strict_eval.get("test_metrics", {}) if isinstance(strict_eval, dict) else {}
if accuracy is None and isinstance(test_metrics, dict):
    accuracy = test_metrics.get("accuracy", None)
if auc is None and isinstance(test_metrics, dict):
    auc = test_metrics.get("auc", None)

profiling = {}
if profiling_path and profiling_path.exists():
    try:
        profiling = json.loads(profiling_path.read_text(encoding="utf-8"))
    except Exception:
        profiling = {}

entry = {
    "dataset": dataset,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "auc": auc,
    "F1": f1,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "AUC": auc,
    "strict_eval": strict_eval,
    "profiling": profiling,
    "log": str(log_path),
}

items = json.loads(summary_json.read_text(encoding="utf-8"))
items.append(entry)
summary_json.write_text(json.dumps(items, indent=2), encoding="utf-8")

with summary_md.open("a", encoding="utf-8") as f:
    acc_s = "NA" if accuracy is None else f"{accuracy:.4f}"
    auc_s = "NA" if auc is None else f"{auc:.4f}"
    f.write(f"| {dataset} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {acc_s} | {auc_s} | {log_path} |\\n")

print(json.dumps(entry, ensure_ascii=False))
PY

  if [[ -n "$profiling_json" && -f "$profiling_json" && -f "$AGGREGATE_PY" ]]; then
    "$PYTHON_BIN" "$AGGREGATE_PY" \
      --baseline "promptem" \
      --dataset "$dataset" \
      --run-id "${TS}_seed${SEED}" \
      --profiling-json "$profiling_json" \
      --summary-json "$SUMMARY_JSON" >/dev/null 2>&1 || true
  fi

  echo "[$(date '+%F %T')] [DATASET=$dataset] DONE" | tee -a "$log_file"
}

for ds in $DATASETS; do
  run_dataset "$ds"
done

echo "Run dir: $RUN_DIR"
echo "Summary: $SUMMARY_MD"
