#!/usr/bin/env bash
set -euo pipefail

# ============== Config ==============
PYTHON_BIN="${PYTHON_BIN:-python}"
SCORE_SCRIPT="${SCORE_SCRIPT:-./score_dataset.py}"

# Where results & logs go
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# Hugging Face auth if your scoring script pulls from HF

# The datasets to score (sequentially)
repo_ids=(
  "IPEC-COMMUNITY/bridge_orig_lerobot"
  "IPEC-COMMUNITY/fractal20220817_data_lerobot"
  "IPEC-COMMUNITY/language_table_lerobot"
  "IPEC-COMMUNITY/fmb_dataset_lerobot"
  "IPEC-COMMUNITY/kuka_lerobot"
  "IPEC-COMMUNITY/droid_lerobot"
  "IPEC-COMMUNITY/bc_z_lerobot"
  "IPEC-COMMUNITY/furniture_bench_dataset_lerobot"
  "IPEC-COMMUNITY/taco_play_lerobot"
  "IPEC-COMMUNITY/berkeley_cable_routing_lerobot"
  "IPEC-COMMUNITY/utaustin_mutex_lerobot"
  "IPEC-COMMUNITY/roboturk_lerobot"
  "IPEC-COMMUNITY/dobbe_lerobot"
  "IPEC-COMMUNITY/stanford_hydra_dataset_lerobot"
  "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"
  "IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot"
  "IPEC-COMMUNITY/berkeley_mvp_lerobot"
  "IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot"
  "IPEC-COMMUNITY/cmu_stretch_lerobot"
  "IPEC-COMMUNITY/berkeley_rpt_lerobot"
  "IPEC-COMMUNITY/viola_lerobot"
  "IPEC-COMMUNITY/dlr_edan_shared_control_lerobot"
  "IPEC-COMMUNITY/jaco_play_lerobot"
  "IPEC-COMMUNITY/toto_lerobot"
  "IPEC-COMMUNITY/cmu_play_fusion_lerobot"
  "IPEC-COMMUNITY/austin_buds_dataset_lerobot"
  "IPEC-COMMUNITY/austin_sirius_dataset_lerobot"
  "IPEC-COMMUNITY/austin_sailor_dataset_lerobot"
  "IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot"
  "IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot"
  "IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot"
)

# ============== Helpers ==============
run_one() {
  local repo_id="$1"

  # Output dir is ./checkpoints/{org}/{name}
  local out_dir="${OUTPUT_ROOT}/${repo_id}"
  local done_flag="${out_dir}/.completed"
  local fail_flag="${out_dir}/.failed"
  local log_file="${LOG_DIR}/$(echo "${repo_id}" | tr '/' '__').log"

  mkdir -p "${out_dir}"

  if [[ -f "${done_flag}" ]]; then
    echo "[SKIP] ${repo_id} (found ${done_flag})"
    return 0
  fi

  echo "[RUN ] ${repo_id} -> ${out_dir}"
  rm -f "${fail_flag}"

  {
    echo "==== $(date -Is) | START ${repo_id} ===="
    echo "CMD: ${PYTHON_BIN} ${SCORE_SCRIPT} --repo_id \"${repo_id}\" --output \"${out_dir}\" --train-baseline=True --train-filtered=True"
    HF_TOKEN="${HF_TOKEN}" "${PYTHON_BIN}" "${SCORE_SCRIPT}" \
      --repo_id "${repo_id}" \
      --root "$HOME/.cache/huggingface/lerobot" \
      --output "${out_dir}" \
      --train-baseline=True \
      --train-filtered=True
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
start_ts=$(date +%s)
echo "Scoring ${#repo_ids[@]} datasets sequentially..."

failed=()

for repo in "${repo_ids[@]}"; do
  if ! run_one "${repo}"; then
    failed+=("${repo}")
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo ""
  echo "Some datasets failed (${#failed[@]}):"
  printf ' - %s\n' "${failed[@]}"
  echo ""
  echo "Failed markers:"
  grep -rl ".failed" "${OUTPUT_ROOT}" || true
  echo "Check logs in: ${LOG_DIR}"
  exit 1
fi

end_ts=$(date +%s)
echo "All scoring runs completed successfully in $(( end_ts - start_ts ))s."

