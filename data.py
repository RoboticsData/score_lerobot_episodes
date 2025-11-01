import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple
import glob
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import os
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import subprocess

V21 = "v2.1"
V30 = "v3.0"

def get_dataset_version(root_path):
    """Detect the dataset version from info.json"""
    info_path = os.path.join(root_path, 'meta/info.json')
    if not os.path.exists(info_path):
        raise ValueError(f"info.json not found at {info_path}")

    with open(info_path, 'r') as f:
        info = json.load(f)

    version = info.get('codebase_version', V21)
    return version

def load_episodes_v30(root_path):
    """Load episodes metadata from v3.0 format (parquet files)"""
    episodes_dir = os.path.join(root_path, 'meta/episodes')
    if not os.path.exists(episodes_dir):
        raise ValueError(f"Episodes directory not found at {episodes_dir}")

    # Find all episode parquet files - v3.0 uses file-*.parquet naming
    parquet_files = sorted(glob.glob(os.path.join(episodes_dir, '**/file-*.parquet'), recursive=True))

    if not parquet_files:
        raise ValueError(f"No episode parquet files found in {episodes_dir}")

    # Read all parquet files and concatenate
    tables = [pq.read_table(f) for f in parquet_files]
    combined_table = pa.concat_tables(tables)
    df = combined_table.to_pandas()

    return df

def get_video_info_v30(df_episodes, episode_idx, camera_key):
    """
    Extract video file info for a specific episode in v3.0 format.

    Args:
        df_episodes: DataFrame with episode metadata
        episode_idx: Episode index to look up
        camera_key: Camera key (can be short form like 'cam_high' or full form like 'observation.images.cam_high')
    """
    ep_row = df_episodes[df_episodes['episode_index'] == episode_idx].iloc[0]

    # Try both short and full camera key formats
    # v3.0 stores full feature names in column headers
    if not camera_key.startswith('observation.images.'):
        full_camera_key = f'observation.images.{camera_key}'
    else:
        full_camera_key = camera_key

    chunk_col = f'videos/{full_camera_key}/chunk_index'
    file_col = f'videos/{full_camera_key}/file_index'
    from_ts_col = f'videos/{full_camera_key}/from_timestamp'
    to_ts_col = f'videos/{full_camera_key}/to_timestamp'

    if chunk_col not in ep_row or file_col not in ep_row:
        return None

    return {
        'chunk_index': int(ep_row[chunk_col]),
        'file_index': int(ep_row[file_col]),
        'from_timestamp': float(ep_row[from_ts_col]),
        'to_timestamp': float(ep_row[to_ts_col]),
    }


def extract_video_segment(video_path, from_timestamp, to_timestamp, output_path=None):
    """
    Extract a segment from a video file using ffmpeg.

    Args:
        video_path: Path to the source video file
        from_timestamp: Start time in seconds
        to_timestamp: End time in seconds
        output_path: Optional output path. If None, creates a temporary file.

    Returns:
        Path to the extracted video segment
    """
    if output_path is None:
        # Create a temporary file
        temp_fd, output_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)

    duration = to_timestamp - from_timestamp

    # Use ffmpeg to extract the segment
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-ss', str(from_timestamp),  # Start time
        '-i', video_path,  # Input file
        '-t', str(duration),  # Duration
        '-c', 'copy',  # Copy codec (fast, no re-encoding)
        '-avoid_negative_ts', '1',  # Handle timestamp issues
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # If copy codec fails, try with re-encoding
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(from_timestamp),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',  # Re-encode video
            '-c:a', 'aac',  # Re-encode audio
            '-avoid_negative_ts', '1',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    return output_path


def get_scorable_video_path(video_path, video_info=None):
    """
    Get a video path suitable for scoring.

    For v2.1: Returns the original path (one episode per video)
    For v3.0: Extracts the episode segment to a temporary file

    Args:
        video_path: Path to the video file
        video_info: Dict with from_timestamp and to_timestamp (v3.0 only)

    Returns:
        tuple: (video_path_for_scoring, is_temporary)
               If is_temporary is True, caller should delete the file after use
    """
    if video_info is None or 'from_timestamp' not in video_info:
        # v2.1 format - return as-is
        return video_path, False

    # v3.0 format - extract segment
    from_ts = video_info['from_timestamp']
    to_ts = video_info['to_timestamp']

    temp_video = extract_video_segment(video_path, from_ts, to_ts)
    return temp_video, True

def update_info_json(info_file):
    # This is required for OpenX datasets since
    # they store the channel name as rgb instead of channels.
    # TODO: Remove this later and fix dataset.
    data = json.load(open(info_file, 'r'))
    for key in data['features']:
        if data['features'][key]['dtype'] != 'video':
            continue
        names = data['features'][key]['names']
        names = ['channels' if x == 'rgb' else x for x in names]
        data['features'][key]['names'] = names
    json.dump(data, open(info_file, 'w'), indent=4)

def load_dataset_hf(repo_id, episodes=None, root=None, revision=None):
    ds_meta = LeRobotDatasetMetadata(
        repo_id, root=root, revision=revision, force_cache_sync=False
    )
    #delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        #delta_timestamps=delta_timestamps,
        #image_transforms=image_transforms,
        #revision=revision,
        #video_backend=cfg.dataset.video_backend,
    )

    # Check and update info.json
    info_file = os.path.join(dataset.root, 'meta/info.json')
    update_info_json(info_file)
    return dataset

def load_jsonl(path):
    assert(path.endswith('.jsonl'))
    with open(path) as f:
        data = [json.loads(line) for line in f]
        return data

def save_jsonl(data, path):
    assert(path.endswith('.jsonl'))
    data = [json.dumps(d) for d in data]
    with open(path, 'w') as f:
        f.write('\n'.join(data))

def rebuild_splits(splits, good_episodes):
    for split in splits:
        start, end = splits[split].split(':')
        start, end = int(start), int(end)
        split_min, split_max = end, start
        for i, ep_idx in enumerate(good_episodes):
            if ep_idx >= start and ep_idx <= end:
                split_min = min(split_min, i)
                split_max = max(split_max, i)
        splits[split] = f"{split_min}:{split_max}"
    return splits

def rewrite_episode_parquet(old_parquet_path, new_parquet_path, good_episodes, start_global_index):
    table = pq.read_table(old_parquet_path)
    n = table.num_rows
    old_episode_idx = table['episode_index'][0].as_py()
    new_episode_idx = good_episodes.index(old_episode_idx)

    # Build/replace columns if present
    def replace_or_add(table, name, array):
        try:
            i = table.schema.get_field_index(name)
            if i != -1:
                return table.set_column(i, name, array)
        except Exception:
            pass
        return table.append_column(name, array)

    # frame_index: 0..n-1
    frame_idx_arr = pa.array(range(n), type=pa.int64())

    # episode_index: constant = new_episode_idx
    episode_idx_arr = pa.array([new_episode_idx] * n, type=pa.int64())

    # global index: start_global_index .. start_global_index + n - 1
    global_idx_arr = pa.array(range(start_global_index, start_global_index + n), type=pa.int64())

    # Only replace existing columns; add if missing (safe for varied schemas)
    if "frame_index" in table.column_names:
        table = replace_or_add(table, "frame_index", frame_idx_arr)
    if "episode_index" in table.column_names:
        table = replace_or_add(table, "episode_index", episode_idx_arr)
    if "index" in table.column_names:
        table = replace_or_add(table, "index", global_idx_arr)

    os.makedirs(os.path.dirname(new_parquet_path), exist_ok=True)
    pq.write_table(table, new_parquet_path, compression="zstd")

    return n  # rows written


def save_filtered_dataset(input_path, output_path, good_episodes, overwrite=True):
    if os.path.exists(input_path) and os.path.exists(output_path) and os.path.samefile(input_path, output_path):
        raise ValueError(f'Input and output path cannot be identical. Input path: {input_path} \nOutput path: {output_path}')
    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(f'Directory {output_path} already exists and overwite is False')
    elif os.path.exists(output_path):
        print(f'Removing directory: {output_path}')
        shutil.rmtree(output_path)

    good_episodes = sorted(list(set(good_episodes)))

    # Detect version and route to appropriate handler
    version = get_dataset_version(input_path)
    print(f"Dataset version detected: {version}")

    if version == V21:
        _save_filtered_dataset_v21(input_path, output_path, good_episodes)
    elif version == V30:
        _save_filtered_dataset_v30(input_path, output_path, good_episodes)
    else:
        raise ValueError(f"Unsupported dataset version: {version}")


def _save_filtered_dataset_v21(input_path, output_path, good_episodes):
    """Filter and save a v2.1 format dataset"""
    # Read meta/info.json
    info_path = os.path.join(input_path, 'meta/info.json')
    info = json.load(open(info_path))

    episode_map = {}
    for new_idx, old_idx in enumerate(good_episodes):
        old_chunk = old_idx // info['chunks_size']
        new_chunk = new_idx // info['chunks_size']
        old_chunk_key = f"chunk-{old_chunk:03d}/episode_{old_idx:06d}"
        new_chunk_key = f"chunk-{new_chunk:03d}/episode_{new_idx:06d}"
        episode_map[old_chunk_key] = new_chunk_key

    # Copy data chunks from data/chunk-*/episode_*.parquet
    # Copy videos from videos/chunk-*/{camera_key}/episode_*.mp4
    camera_keys = list(filter(lambda x: 'images' in x, info['features'].keys()))
    total_videos = 0
    start_global_index = 0
    for old_chunk_key in episode_map:
        new_chunk_key = episode_map[old_chunk_key]

        old_parquet_path = os.path.join(input_path, 'data', old_chunk_key+'.parquet')
        new_parquet_path = os.path.join(output_path, 'data', new_chunk_key+'.parquet')
        os.makedirs(os.path.dirname(new_parquet_path), exist_ok=True)
        shutil.copy2(old_parquet_path, new_parquet_path)

        # Update parquet records.
        n_written_records = rewrite_episode_parquet(
            old_parquet_path,
            new_parquet_path,
            good_episodes,
            start_global_index)
        start_global_index += n_written_records

        for cam in camera_keys:
            old_video_key = os.path.join(old_chunk_key.split('/')[0], cam, old_chunk_key.split('/')[1])+'.mp4'
            new_video_key = os.path.join(new_chunk_key.split('/')[0], cam, new_chunk_key.split('/')[1])+'.mp4'

            old_video_path = os.path.join(input_path, 'videos', old_video_key)
            new_video_path = os.path.join(output_path, 'videos', new_video_key)

            os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
            shutil.copy2(old_video_path, new_video_path)
            total_videos += 1
    assert total_videos > 0, 'Total videos is 0'

    # Copy meta
    os.makedirs(os.path.join(output_path, 'meta'), exist_ok=True)

    # meta/episode_stats.jsonl
    # - only keep lines that have episode_index in episodes
    # - reindex
    episode_stats_input_path = os.path.join(input_path, 'meta/episodes_stats.jsonl')
    episode_stats_output_path = os.path.join(output_path, 'meta/episodes_stats.jsonl')
    episode_stats = load_jsonl(episode_stats_input_path)
    episode_stats = list(filter(lambda x: x['episode_index'] in good_episodes, episode_stats))
    new_episode_stats = []
    for i in range(len(episode_stats)):
        if episode_stats[i]['episode_index'] in good_episodes:
            episode_stats[i]['episode_index'] = good_episodes.index(episode_stats[i]['episode_index'])
            new_episode_stats.append(episode_stats[i])
    save_jsonl(new_episode_stats, episode_stats_output_path)

    # meta/episodes.jsonl
    # - only keep lines that have episode_index in episodes
    # - reindex
    episodes_data_input_path = os.path.join(input_path, 'meta/episodes.jsonl')
    episodes_data_output_path = os.path.join(output_path, 'meta/episodes.jsonl')
    episodes_data = load_jsonl(episodes_data_input_path)
    episodes_data = list(filter(lambda x: x['episode_index'] in good_episodes, episodes_data))
    new_episodes_data = []
    for i in range(len(episodes_data)):
        if episodes_data[i]['episode_index'] in good_episodes:
            episodes_data[i]['episode_index'] = good_episodes.index(episodes_data[i]['episode_index'])
            new_episodes_data.append(episodes_data[i])
    save_jsonl(new_episodes_data, episodes_data_output_path)

    # meta/tasks.jsonl
    # - don't change
    shutil.copy2(os.path.join(input_path, 'meta/tasks.jsonl'), os.path.join(output_path, 'meta/tasks.jsonl'))

    # meta/info.json
    # - update total_episodes
    info['total_episodes'] = len(good_episodes)
    assert info['total_episodes'] > 0, 'Total episodes is 0'
    # - update total_frames
    info['total_frames'] = sum([e['length'] for e in episodes_data])
    assert info['total_frames'] > 0, 'Total frames is 0'
    # - update total_videos
    info['total_videos'] = total_videos
    assert info['total_videos'] > 0, 'Total videos is 0'
    # - update splits
    info['splits'] = rebuild_splits(info['splits'], good_episodes)
    info_output_path = os.path.join(output_path, 'meta/info.json')
    json.dump(info, open(info_output_path, 'w'))


def _save_filtered_dataset_v30(input_path, output_path, good_episodes):
    """
    Filter and save a v3.0 format dataset.

    Note: v3.0 filtering is more complex as episodes are combined in files.
    This implementation uses the LeRobot library to load filtered episodes
    and re-save them.
    """
    import sys
    # Get repo_id from the input path
    # Assume input_path is like: /path/to/cache/repo_owner/repo_name
    path_parts = input_path.rstrip('/').split('/')
    if len(path_parts) >= 2:
        repo_id = f"{path_parts[-2]}/{path_parts[-1]}"
    else:
        raise ValueError(f"Cannot determine repo_id from path: {input_path}")

    print(f"Loading filtered dataset from {repo_id} with episodes: {good_episodes[:5]}...")
    if len(good_episodes) > 5:
        print(f"  ... and {len(good_episodes) - 5} more episodes")

    try:
        # Load only the good episodes using LeRobot's built-in filtering
        filtered_dataset = load_dataset_hf(repo_id, episodes=good_episodes, root=input_path)
    except Exception as e:
        print(f"\nError loading filtered v3.0 dataset: {e}")
        raise


def organize_by_episode(dataset):
    episode_map = {}
    version = get_dataset_version(dataset.root)

    # Get camera keys from features
    camera_keys = [k for k in dataset.meta.features.keys() if 'observation.images' in k]
    camera_keys_clean = [k.replace('observation.images.', '') for k in camera_keys]

    if version == V21:
        # v2.1: Videos organized as videos/chunk-*/CAMERA/episode_*.mp4
        vid_paths = filter(lambda x: '.mp4' in x, dataset.get_episodes_file_paths())

        # Organize videos.
        for vid_path in vid_paths:
            stubs = vid_path.split('/')
            episode_name, camera_type = stubs[-1], stubs[-2]
            print(episode_name)
            episode_idx = int(episode_name.split('_')[1].split('.mp4')[0])
            print(episode_idx)
            if episode_idx not in episode_map:
                episode_map[episode_idx] = {
                    'vid_paths': {},
                    'video_info': {}  # For compatibility
                }
            vid_path = os.path.join(dataset.root, vid_path)
            episode_map[episode_idx]['vid_paths'][camera_type] = vid_path

    elif version == V30:
        # v3.0: Videos organized as videos/CAMERA/chunk-*/file_*.mp4
        # Need to load episodes metadata to find which file each episode is in
        df_episodes = load_episodes_v30(dataset.root)

        # Get unique episode indices from the dataset
        unique_episodes = set()
        for dataset_idx in range(len(dataset)):
            episode_idx = dataset[dataset_idx]["episode_index"].item()
            unique_episodes.add(episode_idx)

        # Build video paths for each episode
        for episode_idx in sorted(unique_episodes):
            if episode_idx not in episode_map:
                episode_map[episode_idx] = {
                    'vid_paths': {},
                    'video_info': {}
                }

            for camera_key in camera_keys_clean:
                video_info = get_video_info_v30(df_episodes, episode_idx, camera_key)
                if video_info is None:
                    continue

                # Build video path: videos/CAMERA/chunk-XXX/file-YYY.mp4
                chunk_idx = video_info['chunk_index']
                file_idx = video_info['file_index']
                vid_path = os.path.join(
                    dataset.root,
                    'videos',
                    f'observation.images.{camera_key}',  # Full feature name
                    f'chunk-{chunk_idx:03d}',
                    f'file-{file_idx:03d}.mp4'  # Note: hyphen, not underscore
                )

                episode_map[episode_idx]['vid_paths'][camera_key] = vid_path
                episode_map[episode_idx]['video_info'][camera_key] = video_info

    else:
        raise ValueError(f"Unsupported dataset version: {version}")

    # Organize actions and states - same for both versions
    # We don't need to load videos at this step.
    for k in camera_keys:
        dataset.meta.features.pop(k, None)

    for dataset_idx in range(len(dataset)):
        timestamp = dataset[dataset_idx]["timestamp"]
        state = dataset[dataset_idx]["observation.state"]
        action = dataset[dataset_idx]["action"]
        episode_idx = dataset[dataset_idx]["episode_index"].item()

        if 'states' not in episode_map[episode_idx]:
            episode_map[episode_idx]['states'] = []

        if 'actions' not in episode_map[episode_idx]:
            episode_map[episode_idx]['actions'] = []

        episode_map[episode_idx]['states'].append(
            {
                "q": state,
                "t": timestamp
            }
        )
        episode_map[episode_idx]['actions'].append(action)

    return episode_map
