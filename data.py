import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple

VID_NAME_RE = re.compile(r"episode_(\d+)\.mp4$")
PARQ_NAME_RE = re.compile(r"episode_(\d+)\.parquet$")

def discover_episodes(root: pathlib.Path, camera: str) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    root = root.expanduser().resolve()
    video_root = root / "videos"
    data_root = root / "data"
    videos = {}
    for chunk in sorted(video_root.glob("chunk-*")):
        cam_dir = chunk / f"observation.images.{camera}"
        if not cam_dir.is_dir(): continue
        for mp4 in cam_dir.glob("episode_*.mp4"):
            m = VID_NAME_RE.match(mp4.name)
            if m: videos[m.group(1)] = mp4
    data = {}
    for chunk in sorted(data_root.glob("chunk-*")):
        for pq in chunk.glob("episode_*.parquet"):
            m = PARQ_NAME_RE.match(pq.name)
            if m: data[m.group(1)] = pq
    common_ids = sorted(set(videos) & set(data))
    return [(videos[i], data[i]) for i in common_ids]

def load_state_from_parquet(pq_path: pathlib.Path) -> Dict[str, np.ndarray]:
    """
    Load Lerobot parquet file with compact `observation.state` array:
        - observation.state           → shape (T, 6), float32
        - observation.eef_pos_x/y/z  → EE position
        - timestamp                  → time
        - observation.contacts       → optional boolean
    """
    df = pd.read_parquet(pq_path)
    state: Dict[str, np.ndarray] = {}

    # timestamps
    if "timestamp" not in df.columns:
        raise KeyError(f"{pq_path}: missing 'timestamp'")
    state["t"] = df["timestamp"].to_numpy(dtype=np.float32)

    # observation.state → joint positions (first 5) + gripper
    if "observation.state" not in df.columns:
        raise KeyError(f"{pq_path}: missing 'observation.state'")
    full_state = np.stack(df["observation.state"].to_numpy())
    state["q"] = full_state[:, :5].astype(np.float32)  # shoulder_pan → wrist_roll
    state["grip"] = full_state[:, 5] > 0.5             # gripper.pos → bool

    return state
