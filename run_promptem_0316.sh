#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/zongze/.venvs/promptem/bin/python}"

DATA_ROOT_BASE="${DATA_ROOT_BASE:-/home/mengshi/table_quality/datasets_joint_discovery_integration}"
DATASETS="${DATASETS:-wikidbs_1218 santos_benchmark_1218 magellan_1218}"

# Training defaults (full-label EM)
DEVICE="${DEVICE:-cuda}"
K="${K:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-2e-5}"
NUM_ITER="${NUM_ITER:-1}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-30}"
TEST_EVERY="${TEST_EVERY:-1}"
SEED="${SEED:-2022}"
ONLY_PLM="${ONLY_PLM:-1}"
SELF_TRAIN="${SELF_TRAIN:-0}"
DYNAMIC_DATASET="${DYNAMIC_DATASET:--1}"
UNCERTAINTY_RATIO="${UNCERTAINTY_RATIO:-0.1}"
EL2N_RATIO="${EL2N_RATIO:-0.1}"
MC_DROPOUT_PASS="${MC_DROPOUT_PASS:-10}"
TEMPLATE_NO="${TEMPLATE_NO:-0}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"

# Optional per-dataset overrides to speed up large datasets.
MAGELLAN_TEACHER_EPOCHS="${MAGELLAN_TEACHER_EPOCHS:-10}"
MAGELLAN_STUDENT_EPOCHS="${MAGELLAN_STUDENT_EPOCHS:-30}"
MAGELLAN_TEST_EVERY="${MAGELLAN_TEST_EVERY:-10}"

RUNS_ROOT="${RUNS_ROOT:-/home/zongze/mengshichen_projects/runs/promptem_em}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_ROOT}/k1p0_${TS}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_MD="${RUN_DIR}/summary.md"

cat > "$SUMMARY_MD" <<'MD'
| Dataset | Precision | Recall | F1 | Accuracy | AUC | Log |
| ------- | --------: | -----: | --:| -------: | --: | --- |
MD

"$PYTHON" - <<PY
import json
from pathlib import Path
Path("$SUMMARY_JSON").write_text(json.dumps([], indent=2), encoding="utf-8")
PY

run_dataset() {
  local dataset="$1"
  local log_file="$LOG_DIR/${dataset}.log"
  local dataset_root="$DATA_ROOT_BASE/$dataset"
  local promptem_data_dir="$ROOT_DIR/data/$dataset"
  local local_teacher_epochs="$TEACHER_EPOCHS"
  local local_student_epochs="$STUDENT_EPOCHS"
  local local_test_every="$TEST_EVERY"

  if [[ "$dataset" == "magellan_1218" ]]; then
    local_teacher_epochs="$MAGELLAN_TEACHER_EPOCHS"
    local_student_epochs="$MAGELLAN_STUDENT_EPOCHS"
    local_test_every="$MAGELLAN_TEST_EVERY"
  fi

  echo "[$(date '+%F %T')] [DATASET=$dataset] START" | tee "$log_file"

  if [[ ! -d "$dataset_root" ]]; then
    echo "[ERROR] dataset root not found: $dataset_root" | tee -a "$log_file"
    return 1
  fi

  if [[ "$FORCE_CONVERT" == "1" || ! -f "$promptem_data_dir/manifest.json" ]]; then
    echo "[$(date '+%F %T')] [DATASET=$dataset] convert 1218 -> PromptEM" | tee -a "$log_file"
    "$PYTHON" "$ROOT_DIR/convert_1218_to_promptem.py" \
      --dataset-root "$dataset_root" \
      --output-dir "$promptem_data_dir" \
      --max-cell-chars 200 \
      --skip-empty 2>&1 | tee -a "$log_file"
  else
    echo "[$(date '+%F %T')] [DATASET=$dataset] skip convert (manifest exists)" | tee -a "$log_file"
  fi

  cmd=(
    "$PYTHON" "$ROOT_DIR/main.py"
    -d "$dataset"
    --device "$DEVICE"
    -k "$K"
    -bs "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --lr "$LR"
    -ni "$NUM_ITER"
    -te "$local_teacher_epochs"
    -se "$local_student_epochs"
    --test_every "$local_test_every"
    -ur "$UNCERTAINTY_RATIO"
    -er "$EL2N_RATIO"
    -mdp "$MC_DROPOUT_PASS"
    -tn "$TEMPLATE_NO"
    --seed "$SEED"
  )

  if [[ "$SELF_TRAIN" == "1" ]]; then
    cmd+=(-st)
  fi

  if [[ "$DYNAMIC_DATASET" != "-1" ]]; then
    cmd+=(-dd "$DYNAMIC_DATASET")
  fi

  if [[ "$ONLY_PLM" == "1" ]]; then
    cmd+=(--only_plm)
  fi

  echo "[$(date '+%F %T')] [DATASET=$dataset] run: ${cmd[*]}" | tee -a "$log_file"
  "${cmd[@]}" 2>&1 | tee -a "$log_file"

  "$PYTHON" - "$dataset" "$log_file" "$SUMMARY_MD" "$SUMMARY_JSON" <<'PY'
import json
import re
import sys
from pathlib import Path

dataset = sys.argv[1]
log_path = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
summary_json = Path(sys.argv[4])

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

entry = {
    "dataset": dataset,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "auc": auc,
    "log": str(log_path),
}

items = json.loads(summary_json.read_text(encoding="utf-8"))
items.append(entry)
summary_json.write_text(json.dumps(items, indent=2), encoding="utf-8")

with summary_md.open("a", encoding="utf-8") as f:
    acc_s = "NA" if accuracy is None else f"{accuracy:.4f}"
    auc_s = "NA" if auc is None else f"{auc:.4f}"
    f.write(f"| {dataset} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {acc_s} | {auc_s} | {log_path} |\n")

print(json.dumps(entry, ensure_ascii=False))
PY

  echo "[$(date '+%F %T')] [DATASET=$dataset] DONE" | tee -a "$log_file"
}

for ds in $DATASETS; do
  run_dataset "$ds"
done

echo "Run dir: $RUN_DIR"
echo "Summary: $SUMMARY_MD"
