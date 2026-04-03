#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT_BASE="${DATA_ROOT_BASE:-/home/zongze/mengshichen_projects/datasets_joint_discovery_integration_split_work}"
SPLIT_TAG="${SPLIT_TAG:-040303}"
DATASETS="${DATASETS:-wikidbs santos_benchmark magellan}"
RUNS_ROOT="${RUNS_ROOT:-$SCRIPT_DIR/runs/promptem_em_split_${SPLIT_TAG}}"
STRICT_EVAL="${STRICT_EVAL:-1}"
PROFILE="${PROFILE:-0}"
PROFILE_INTERVAL_SEC="${PROFILE_INTERVAL_SEC:-1.0}"
PROFILE_OUT="${PROFILE_OUT:-}"

normalize_dataset() {
  case "$1" in
    wikidbs|wikidbs_*) echo "wikidbs_${SPLIT_TAG}" ;;
    santos|santos_benchmark|santos_benchmark_*) echo "santos_benchmark_${SPLIT_TAG}" ;;
    magellan|magellan_*) echo "magellan_${SPLIT_TAG}" ;;
    *) echo "$1" ;;
  esac
}

datasets_norm=()
for ds in $DATASETS; do
  datasets_norm+=("$(normalize_dataset "$ds")")
done

DATASETS_JOINED="${datasets_norm[*]}"

for ds in ${DATASETS_JOINED}; do
  ds_root="${DATA_ROOT_BASE}/${ds}"
  for req in \
    "${ds_root}" \
    "${ds_root}/datalake_plus" \
    "${ds_root}/label_plus/entity_matching/train.csv" \
    "${ds_root}/label_plus/entity_matching/validate.csv" \
    "${ds_root}/label_plus/entity_matching/test.csv"; do
    if [[ ! -e "${req}" ]]; then
      echo "[ERROR] Missing required path for ${ds}: ${req}" >&2
      exit 2
    fi
  done
done

echo "========================================="
echo " PromptEM Split Run"
echo " DATA_ROOT_BASE=$DATA_ROOT_BASE"
echo " SPLIT_TAG=$SPLIT_TAG"
echo " DATASETS=$DATASETS_JOINED"
echo " RUNS_ROOT=$RUNS_ROOT"
echo "========================================="

DATA_ROOT_BASE="$DATA_ROOT_BASE" \
DATASETS="$DATASETS_JOINED" \
RUNS_ROOT="$RUNS_ROOT" \
STRICT_EVAL="$STRICT_EVAL" \
PROFILE="$PROFILE" \
PROFILE_INTERVAL_SEC="$PROFILE_INTERVAL_SEC" \
PROFILE_OUT="$PROFILE_OUT" \
"${SCRIPT_DIR}/run_promptem_0316.sh" "$@"
