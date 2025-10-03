import argparse
import os
import cv2
import numpy as np
import json
import shutil
import random
from typing import List, Dict
import pyarrow as pa
import pyarrow.parquet as pq
from data import load_dataset_hf, organize_by_episode


def corrupt_video_frame(frame: np.ndarray, corruption_strength: float = 0.5) -> np.ndarray:
    """Apply visual corruption proportional to corruption_strength (0–1)."""
    corrupted_frame = frame.copy()

    # ---- Blur proportional to strength ----
    if corruption_strength > 0:
        # Interpolate kernel size between 1 (no blur) and 11 (strong blur), must be odd
        max_kernel = 11
        kernel_size = int(1 + corruption_strength * (max_kernel - 1))
        if kernel_size % 2 == 0:  # kernel must be odd
            kernel_size += 1
        if kernel_size > 1:  # apply only if >1
            corrupted_frame = cv2.GaussianBlur(corrupted_frame, (kernel_size, kernel_size), 0)

    # ---- Lighting variation proportional to strength ----
    if corruption_strength > 0:
        max_brightness = 50
        max_contrast_delta = 0.5

        brightness = corruption_strength * random.uniform(-max_brightness, max_brightness)
        contrast   = 1.0 + corruption_strength * random.uniform(-max_contrast_delta, max_contrast_delta)

        corrupted_frame = corrupted_frame.astype(np.float32)
        corrupted_frame = corrupted_frame * contrast + brightness
        corrupted_frame = np.clip(corrupted_frame, 0, 255).astype(np.uint8)

    return corrupted_frame


def corrupt_video(input_path: str, output_path: str, corruption_prob: float) -> bool:
    """
    Corrupt an entire video file, writing to a DIFFERENT output path.
    Returns True if corruption was applied and output written, False otherwise.
    """
    # Don't corrupt this video with (1 - p)
    if random.random() >= corruption_prob:
        return False

    # Must not write to the same file we're reading
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("output_path must be different from input_path")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not open VideoWriter for: {output_path}")

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally enable visual corruption:
        frame = corrupt_video_frame(frame, corruption_strength=0.7)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Corrupted video saved: {output_path} ({frame_count}/{total_frames} frames processed)")
    return True

def corrupt_motion_data(states: List[Dict], actions: List[np.ndarray], corruption_prob: float) -> tuple:
    """Add noise to motion data (states and actions)."""
    corrupted_states = []
    corrupted_actions = []

    # First decide if this episode should be corrupted at all
    if random.random() >= corruption_prob:
        # No corruption for this episode
        for state_dict, action in zip(states, actions):
            corrupted_states.append(state_dict.copy())
            corrupted_actions.append(action.clone())
        return corrupted_states, corrupted_actions, False, False

    # Episode is selected for corruption - determine corruption amount
    # Base corruption: 20% of timesteps, with random variation up to 80%
    base_corruption_rate = 0.2
    max_additional_corruption = 0.6
    timestep_corruption_rate = base_corruption_rate + random.random() * max_additional_corruption

    state_corrupted = False
    action_corrupted = False

    for i, (state_dict, action) in enumerate(zip(states, actions)):
        corrupted_state = state_dict.copy()
        corrupted_action = action.clone()

        # Corrupt state data based on timestep corruption rate
        if random.random() < timestep_corruption_rate:
            state_array = state_dict["q"]
            noise_scale = 0.05  # 5% noise
            noise = np.random.normal(0, noise_scale, state_array.shape)
            corrupted_state["q"] = state_array + noise
            state_corrupted = True

        # Corrupt action data based on timestep corruption rate
        if random.random() < timestep_corruption_rate:
            noise_scale = 0.03  # 3% noise for actions
            noise = np.random.normal(0, noise_scale, action.shape)
            corrupted_action = action + noise
            action_corrupted = True

        corrupted_states.append(corrupted_state)
        corrupted_actions.append(corrupted_action)

    return corrupted_states, corrupted_actions, state_corrupted, action_corrupted


def update_parquet_with_corrupted_data(parquet_path: str, corrupted_states: List[Dict], corrupted_actions: List[np.ndarray]):
    """Update parquet file with corrupted motion data."""
    table = pq.read_table(parquet_path)
    
    # Extract existing data
    data_dict = {}
    for column_name in table.column_names:
        data_dict[column_name] = table[column_name].to_pylist()
    
    # Update state and action columns
    for i, (state_dict, action) in enumerate(zip(corrupted_states, corrupted_actions)):
        if i < len(data_dict.get("observation.state", [])):
            # Convert to numpy array if it's a tensor
            state_q = state_dict["q"]
            if hasattr(state_q, 'numpy'):
                state_q = state_q.numpy()
            elif hasattr(state_q, 'detach'):
                state_q = state_q.detach().numpy()
            # Ensure consistent float32 dtype
            if isinstance(state_q, np.ndarray):
                state_q = state_q.astype(np.float32)
            data_dict["observation.state"][i] = state_q
        if i < len(data_dict.get("action", [])):
            # Convert to numpy array if it's a tensor
            action_data = action
            if hasattr(action_data, 'numpy'):
                action_data = action_data.numpy()
            elif hasattr(action_data, 'detach'):
                action_data = action_data.detach().numpy()
            # Ensure consistent float32 dtype
            if isinstance(action_data, np.ndarray):
                action_data = action_data.astype(np.float32)
            data_dict["action"][i] = action_data
    
    # Create new table with corrupted data
    new_table = pa.table(data_dict)
    pq.write_table(new_table, parquet_path, compression="zstd")


def corrupt_dataset(repo_id: str, output_path: str, corruption_prob: float, overwrite: bool = True, root=None):
    """Corrupt an entire dataset by applying corruption to videos and motion data."""

    if os.path.exists(output_path):
        if overwrite:
            print(f'Removing existing directory: {output_path}')
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f'Directory {output_path} already exists and overwrite is False')

    # Copy entire dataset structure first
    dataset = load_dataset_hf(repo_id, root=root)
    print("Copying dataset structure...")
    shutil.copytree(dataset.root, output_path)
    episode_map = organize_by_episode(dataset)

    # Track corruption information
    corruption_log = {
        "corruption_probability": corruption_prob,
        "corrupted_episodes": {},
        "total_episodes": len(episode_map),
        "corruption_types": {
            "video_corruption_strength": 0.7,
            "state_noise_scale": 0.05,
            "action_noise_scale": 0.03,
            "timestep_corruption": {
                "base_rate": 0.2,
                "max_additional": 0.6,
                "description": "For corrupted episodes, 20-80% of timesteps are corrupted"
            }
        }
    }

    # Corrupt videos
    print(f"Corrupting videos with {corruption_prob}% corruption proportion...")
    video_dir = os.path.join(output_path, 'videos')
    if os.path.exists(video_dir):
        for root_dir, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root_dir, file)
                    print(f"Processing video: {video_path}")

                    # Extract episode index from video path
                    episode_idx = None
                    if 'episode_' in file:
                        try:
                            episode_idx = int(file.split('episode_')[1].split('.')[0])
                        except (IndexError, ValueError):
                            pass
                    # Write to a temp path first, then atomically replace on success
                    tmp_out = f"{video_path}.tmp_corrupted.mp4"
                    video_corrupted = False
                    try:
                        video_corrupted = corrupt_video(video_path, tmp_out, corruption_prob)
                        if video_corrupted:
                            os.replace(tmp_out, video_path)  # atomic on same filesystem
                        else:
                            # No corruption applied: ensure temp is removed if created
                            if os.path.exists(tmp_out):
                                os.remove(tmp_out)
                    finally:
                        # Extra cleanup in case of exceptions
                        if os.path.exists(tmp_out) and not video_corrupted:
                            try:
                                os.remove(tmp_out)
                            except OSError as e:
                                print(e)
                                pass

                    if episode_idx is not None:
                        if episode_idx not in corruption_log["corrupted_episodes"]:
                            corruption_log["corrupted_episodes"][episode_idx] = {
                                "video_corrupted": False,
                                "state_corrupted": False,
                                "action_corrupted": False
                            }
                        corruption_log["corrupted_episodes"][episode_idx]["video_corrupted"] = video_corrupted

    # Corrupt motion data in parquet files
    print(f"Corrupting motion data with {corruption_prob}% corruption proportion...")
    data_dir = os.path.join(output_path, 'data')
    if os.path.exists(data_dir):
        for root_dir, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_path = os.path.join(root_dir, file)

                    # Extract episode index from filename
                    episode_idx = None
                    if 'episode_' in file:
                        try:
                            episode_idx = int(file.split('episode_')[1].split('.')[0])
                        except (IndexError, ValueError):
                            pass

                    # Get episode data if available
                    if episode_idx is not None and episode_idx in episode_map:
                        episode = episode_map[episode_idx]
                        states = episode.get('states', [])
                        actions = episode.get('actions', [])

                        if states and actions:
                            print(f"Corrupting motion data for episode {episode_idx}")
                            corrupted_states, corrupted_actions, state_corrupted, action_corrupted = corrupt_motion_data(
                                states, actions, corruption_prob
                            )
                            update_parquet_with_corrupted_data(
                                parquet_path, corrupted_states, corrupted_actions
                            )

                            # Track corruption
                            if episode_idx not in corruption_log["corrupted_episodes"]:
                                corruption_log["corrupted_episodes"][episode_idx] = {
                                    "video_corrupted": False,
                                    "state_corrupted": False,
                                    "action_corrupted": False
                                }
                            corruption_log["corrupted_episodes"][episode_idx]["state_corrupted"] = state_corrupted
                            corruption_log["corrupted_episodes"][episode_idx]["action_corrupted"] = action_corrupted
                    else:
                        print(f"Skipping motion corruption for {parquet_path} (no episode data)")

    # Save corruption log
    corruption_log_path = os.path.join(output_path, "corruption_log.json")
    with open(corruption_log_path, 'w') as f:
        json.dump(corruption_log, f, indent=2)

    # Print summary
    total_corrupted = len(corruption_log["corrupted_episodes"])
    video_corrupted = sum(1 for ep in corruption_log["corrupted_episodes"].values() if ep["video_corrupted"])
    state_corrupted = sum(1 for ep in corruption_log["corrupted_episodes"].values() if ep["state_corrupted"])
    action_corrupted = sum(1 for ep in corruption_log["corrupted_episodes"].values() if ep["action_corrupted"])

    print(f"\nCorruption Summary:")
    print(f"  Total episodes processed: {corruption_log['total_episodes']}")
    print(f"  Episodes with any corruption: {total_corrupted}")
    print(f"  Episodes with video corruption: {video_corrupted}")
    print(f"  Episodes with state corruption: {state_corrupted}")
    print(f"  Episodes with action corruption: {action_corrupted}")
    print(f"  Corruption log saved to: {corruption_log_path}")

    print(f"Dataset corruption complete. Corrupted dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Corrupt HuggingFace LeRobot dataset with video and motion corruption")
    parser.add_argument("--repo_id", required=True, help="Repository ID of the dataset")
    parser.add_argument("--root", default=None, help="Root directory of the dataset")
    parser.add_argument("--corruption_prob", type=float, default=0.5, 
                       help="Proportion of episodes to corrupt (0-1)")
    parser.add_argument("--output_path", default=None,
                       help="Custom suffix for output directory (default: _corrupted_X%%)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                       help="Overwrite output directory if it exists")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42) (set to -1 to disable seeding)")
    
    args = parser.parse_args()
    
    # Generate output path
    output_path = args.output_path
    repo_id = args.repo_id
    root = args.root
    
    print(f"Input repo_id: {repo_id}")
    print(f"Input root: {root}")
    print(f"Output dataset: {output_path}")
    print(f"Corruption proportion: {args.corruption_prob}%")

    #set random seed for reproducibility, we need to set the seed for both random and np.random
    if args.seed != -1:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed) #we use random in helper function to corrupt video frames
        np.random.seed(args.seed) #we use np.random to corrupt motion data
        
    # Validate corruption proportion
    if not 0 <= args.corruption_prob <= 1:
        raise ValueError("Corruption proportion must be between 0 and 1")
    
    # Corrupt the dataset
    corrupt_dataset(repo_id, output_path, args.corruption_prob, args.overwrite, root=root)


if __name__ == "__main__":
    main()