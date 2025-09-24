#!/usr/bin/env bash
set -euo pipefail

# ============== Config ==============
PYTHON_BIN="${PYTHON_BIN:-python}"
CORRUPT_SCRIPT="${CORRUPT_SCRIPT:-./corrupt.py}"

# Where results & logs go
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/local/output_corrupt}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_corrupt_20}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# Parallelism
PARALLEL_JOBS="${PARALLEL_JOBS:-10}"

# HF auth if needed
export HF_TOKEN="${HF_TOKEN:-}"

# Extra args to pass to corrupt.py
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ============== Dataset list (copied from original) ==============
repo_ids=(
  "IPEC-COMMUNITY/austin_buds_dataset_lerobot"
  "IPEC-COMMUNITY/austin_sirius_dataset_lerobot"
  "IPEC-COMMUNITY/austin_sailor_dataset_lerobot"
  "IPEC-COMMUNITY/utaustin_mutex_lerobot"
  "IPEC-COMMUNITY/dlr_edan_shared_control_lerobot"
  "IPEC-COMMUNITY/cmu_stretch_lerobot"
  "IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot"
  "IPEC-COMMUNITY/viola_lerobot"
  "IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot"
  "IPEC-COMMUNITY/austin_sailor_dataset_lerobot"
  "IPEC-COMMUNITY/berkeley_mvp_lerobot"
  "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"
  "IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot"
  "IPEC-COMMUNITY/berkeley_rpt_lerobot"
  "IPEC-COMMUNITY/cmu_play_fusion_lerobot"
  "IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot"
  "IPEC-COMMUNITY/toto_lerobot"
  "IPEC-COMMUNITY/kuka_lerobot"
  "IPEC-COMMUNITY/droid_lerobot"
  "IPEC-COMMUNITY/berkeley_cable_routing_lerobot"
  "IPEC-COMMUNITY/stanford_hydra_dataset_lerobot"
  "IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot"
  #"Daddyboss/so_100_test4"
  #"IPEC-COMMUNITY/bridge_orig_lerobot"
  #"IPEC-COMMUNITY/fractal20220817_data_lerobot"
  #"IPEC-COMMUNITY/language_table_lerobot"
  #"IPEC-COMMUNITY/fmb_dataset_lerobot"
  #"IPEC-COMMUNITY/bc_z_lerobot"
  #"IPEC-COMMUNITY/furniture_bench_dataset_lerobot"
  #"IPEC-COMMUNITY/taco_play_lerobot"
  #"IPEC-COMMUNITY/roboturk_lerobot"
  #"IPEC-COMMUNITY/dobbe_lerobot"
  #"IPEC-COMMUNITY/dlr_edan_shared_control_lerobot"
  #"IPEC-COMMUNITY/jaco_play_lerobot"
)

# ============== Helpers ==============
run_one() {
  local repo_id="$1"
  local out_dir="${OUTPUT_ROOT}/${repo_id}${OUTPUT_SUFFIX}"
  local done_flag="${out_dir}/.completed"
  local fail_flag="${out_dir}/.failed"
  local log_file="${LOG_DIR}/$(echo "${repo_id}${OUTPUT_SUFFIX}" | tr '/' '__').log"

  if [[ -f "${done_flag}" ]]; then
    echo "[SKIP] ${repo_id}"
    return 0
  fi

  echo "[RUN ] ${repo_id} -> ${out_dir}"
  rm -f "${fail_flag}"

  {
    echo "==== $(date -Is) | START ${repo_id} ===="
    echo "CMD: ${PYTHON_BIN} ${CORRUPT_SCRIPT} --repo_id \"${repo_id}\" --output \"${out_dir}\" ${EXTRA_ARGS}"
    HF_TOKEN="${HF_TOKEN}" "${PYTHON_BIN}" "${CORRUPT_SCRIPT}" \
      --repo_id "${repo_id}" \
      --output "${out_dir}" \
      --train_filtered True \
      --train_baseline True \
      ${EXTRA_ARGS}
    echo "==== $(date -Is) | DONE  ${repo_id} ===="
  } > "${log_file}" 2>&1 || {
    echo "[FAIL] ${repo_id} (see ${log_file})"
    printf "failed at %s\n" "$(date -Is)" > "${fail_flag}"
    return 1
  }

  touch "${done_flag}"
  echo "[OK  ] ${repo_id}"
}

# ============== Main ==============
echo "Corrupting ${#repo_ids[@]} dataset(s) with PARALLEL_JOBS=${PARALLEL_JOBS}"
active=0
rc=0

for repo in "${repo_ids[@]}"; do
  run_one "${repo}" &
  ((active+=1))
  if (( active >= PARALLEL_JOBS )); then
    wait -n || rc=1
    ((active-=1))
  fi
done

wait || rc=1

if (( rc != 0 )); then
  echo ""
  echo "Some runs failed. Failed markers:"
  grep -rl ".failed" "${OUTPUT_ROOT}" || true
  echo "Check logs in: ${LOG_DIR}"
  exit 1
fi

echo "All corruption runs completed successfully."

