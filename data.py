import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple
import glob
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import os
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

def load_dataset_hf(repo_id, episodes=None, root=None, revision=None):
    ds_meta = LeRobotDatasetMetadata(
        repo_id, root=root, revision=revision, force_cache_sync=False
    )
    #delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        #episodes=episodes,
        #delta_timestamps=delta_timestamps,
        #image_transforms=image_transforms,
        #revision=revision,
        #video_backend=cfg.dataset.video_backend,
    )
    camera_keys = ds_meta.camera_keys
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

def organize_by_episode(dataset):
    episode_map = {}
    vid_paths = filter(lambda x: '.mp4' in x, dataset.get_episodes_file_paths())

    # Organize videos.
    for vid_path in vid_paths:
        stubs = vid_path.split('/')
        episode_name, camera_type = stubs[-1], stubs[-2]
        episode_idx = int(episode_name.split('_')[1].split('.mp4')[0])
        if episode_idx not in episode_map:
            episode_map[episode_idx] = {
                'vid_paths' : {}
            }
        vid_path = os.path.join(dataset.root, vid_path)
        episode_map[episode_idx]['vid_paths'][camera_type] = vid_path
 
    # Organize actions.
    # We don't need to load videos at this step.
    camera_keys = filter(lambda x: 'observation.images' in x, list(dataset.meta.features.keys()))
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
