#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
HFD="./hfd.sh"                          # path to your downloader script
LOCAL_DIR="${HOME}/.cache/huggingface/lerobot"
BATCH_SIZE=10

# List of repo_ids
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

ARIA2C_OPTS="\
  --continue=true \
  --max-concurrent-downloads=4 \
  --split=4 \
  --min-split-size=64M \
  --retry-wait=5 \
  --max-tries=0 \
  --conditional-get=true \
  --auto-file-renaming=true \
  --summary-interval=0"

mkdir -p "$LOCAL_DIR"

#HF_TOKEN="$HF_TOKEN"
# --- Batched parallel downloads ---
for ((i=0; i<${#repo_ids[@]}; i+=BATCH_SIZE)); do
  for ((j=i; j<i+BATCH_SIZE && j<${#repo_ids[@]}; j++)); do
    repo_id="${repo_ids[$j]}"
    echo "Downloading $repo_id ..."
    HF_HUB_ENABLE_HF_TRANSFER=1 \
      "$HFD" "$repo_id" --dataset --local-dir "$LOCAL_DIR/$repo_id" &
  done
  wait  # wait for this batch to complete before starting the next
done

echo "All downloads completed!"

