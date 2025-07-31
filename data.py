import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple
import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

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
    print(dataset.get_episodes_file_paths())
    # TODO: Discover episodes (videos and parq).
    # Output state and actions
    print(dataset[0]["action"])
    print(dataset[0]["observation.state"])
    return dataset
