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


def corrupt_video(input_path: str, output_path: str, corruption_prob: float):
    """Corrupt an entire video file."""

    # Don't corrupt this video with 1-p
    if random.random() >= corruption_prob:
        return
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Corrupt current frame
        frame = corrupt_video_frame(frame, corruption_strength=0.7)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Corrupted video saved: {output_path} ({frame_count}/{total_frames} frames processed)")


def corrupt_motion_data(states: List[Dict], actions: List[np.ndarray], corruption_prob: float) -> tuple:
    """Add noise to motion data (states and actions)."""
    corrupted_states = []
    corrupted_actions = []
    
    for i, (state_dict, action) in enumerate(zip(states, actions)):
        corrupted_state = state_dict.copy()
        corrupted_action = action.clone()
        
        # Corrupt state data
        if random.random() < corruption_prob:
            state_array = state_dict["q"]
            noise_scale = 0.05  # 5% noise
            noise = np.random.normal(0, noise_scale, state_array.shape)
            corrupted_state["q"] = state_array + noise
        
        # Corrupt action data
        if random.random() < corruption_prob:
            noise_scale = 0.03  # 3% noise for actions
            noise = np.random.normal(0, noise_scale, action.shape)
            corrupted_action = action + noise
        
        corrupted_states.append(corrupted_state)
        corrupted_actions.append(corrupted_action)
    
    return corrupted_states, corrupted_actions


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
            data_dict["observation.state"][i] = state_dict["q"]
        if i < len(data_dict.get("action", [])):
            data_dict["action"][i] = action
    
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
        
    # Corrupt videos
    print(f"Corrupting videos with {corruption_prob}% corruption proportion...")
    video_dir = os.path.join(output_path, 'videos')
    if os.path.exists(video_dir):
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    print(f"Processing video: {video_path}")
                    corrupt_video(video_path, video_path, corruption_prob)
    
    # Corrupt motion data in parquet files
    print(f"Corrupting motion data with {corruption_prob}% corruption proportion...")
    data_dir = os.path.join(output_path, 'data')
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_path = os.path.join(root, file)
                    
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
                            corrupted_states, corrupted_actions = corrupt_motion_data(
                                states, actions, corruption_prob
                            )
                            update_parquet_with_corrupted_data(
                                parquet_path, corrupted_states, corrupted_actions
                            )
                    else:
                        print(f"Skipping motion corruption for {parquet_path} (no episode data)")
    
    print(f"Dataset corruption complete. Corrupted dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Corrupt HuggingFace LeRobot dataset with video and motion corruption")
    parser.add_argument("--repo_id", required=True, help="Repository ID of the dataset")
    parser.add_argument("--root", default=None, help="Root directory of the dataset")
    parser.add_argument("--corruption_prob", type=float, default=0.2, 
                       help="Proportion of episodes to corrupt (0-1)")
    parser.add_argument("--output_path", default=None, 
                       help="Custom suffix for output directory (default: _corrupted_X%)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                       help="Overwrite output directory if it exists")
    
    args = parser.parse_args()
    
    # Generate output path
    output_path = args.output_path
    repo_id = args.repo_id
    root = args.root
    
    print(f"Input repo_id: {repo_id}")
    print(f"Input root: {root}")
    print(f"Output dataset: {output_path}")
    print(f"Corruption proportion: {args.corruption_prob}%")
    
    # Validate corruption proportion
    if not 0 <= args.corruption_prob <= 1:
        raise ValueError("Corruption proportion must be between 0 and 1")
    
    # Corrupt the dataset
    corrupt_dataset(repo_id, output_path, args.corruption_prob, args.overwrite, root=root)


if __name__ == "__main__":
    main()