import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple
import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import os

def load_dataset_hf(repo_id, root=None, revision=None):
    ds_meta = LeRobotDatasetMetadata(
        repo_id, root=root, revision=revision
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
