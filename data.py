import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple
import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import os
import json
import shutil

def load_dataset_hf(repo_id, episodes=None, root=None, revision=None):
    ds_meta = LeRobotDatasetMetadata(
        repo_id, root=root, revision=revision, force_cache_sync=True
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
        split_min, split_max = start, end
        for ep_idx in good_episodes:
            if ep_idx >= start and ep_idx <= end:
                split_min = min(split_min, ep_idx)
                split_max = max(split_max, ep_idx)
        splits[split] = f"{split_min}:{split_max}"
    return splits


def save_filtered_dataset(input_path, output_path, good_episodes):
    good_episodes = sorted(good_episodes)
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
    for old_chunk_key in episode_map:
        new_chunk_key = episode_map[old_chunk_key]

        old_parquet_path = os.path.join(input_path, 'data', old_chunk_key+'.parquet')
        new_parquet_path = os.path.join(output_path, 'data', new_chunk_key+'.parquet')
        os.makedirs(os.path.dirname(new_parquet_path), exist_ok=True)
        shutil.copy2(old_parquet_path, new_parquet_path)

        for cam in camera_keys:
            old_video_key = os.path.join(old_chunk_key.split('/')[0], cam, old_chunk_key.split('/')[1])+'.mp4'
            new_video_key = os.path.join(new_chunk_key.split('/')[0], cam, new_chunk_key.split('/')[1])+'.mp4'

            old_video_path = os.path.join(input_path, 'videos', old_video_key)
            new_video_path = os.path.join(output_path, 'videos', new_video_key)

            os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
            shutil.copy2(old_video_path, new_video_path)
            total_videos += 1

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
            # TODO: Fix this, append to new_episode_stats
            episode_stats[i]['episode_index'] = good_episodes.index(episode_stats[i]['episode_index'])
    save_jsonl(new_episode_stats, episode_stats_output_path)

    # meta/episodes.jsonl
    # - only keep lines that have episode_index in episodes
    # - reindex
    episodes_data_input_path = os.path.join(input_path, 'meta/episodes.jsonl')
    episodes_data_output_path = os.path.join(output_path, 'meta/episodes.jsonl')
    episodes_data = load_jsonl(episodes_data_input_path)
    episodes_data = list(filter(lambda x: x['episode_index'] in good_episodes, episodes_data))
    for i in range(len(episodes_data)):
        if episodes_data[i]['episode_index'] in good_episodes:
            episodes_data[i]['episode_index'] = good_episodes.index(episodes_data[i]['episode_index'])
    save_jsonl(new_episodes_data, episodes_data_output_path)

    # meta/tasks.jsonl
    # - don't change
    shutil.copy2(os.path.join(input_path, 'meta/tasks.jsonl'), os.path.join(output_path, 'meta/tasks.jsonl'))

    # meta/info.json
    # - update total_episodes
    info['total_episodes'] = len(good_episodes)
    # - update total_frames
    info['total_frames'] = sum([e['length'] for e in episodes_data])
    # - update total_videos
    info['total_videos'] = total_videos
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
