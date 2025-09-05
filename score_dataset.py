import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple

from vlm import VLMInterface
from data import organize_by_episode, load_dataset_hf, save_filtered_dataset
from scores import score_task_success, score_visual_clarity, score_smoothness, score_path_efficiency, score_collision, score_runtime, score_joint_stability, score_gripper_consistency
from scores import build_time_stats           # (your helper from the other file)
from train import start_training
import hashlib
import pickle
import os
import uniplot
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import train as lerobot_train
from lerobot.constants import HF_LEROBOT_HOME

class DatasetScorer:
    def __init__(self, vlm: VLMInterface, time_stats: dict):
        def runtime_with_stats(vp, st, vlm, task, nominal):
            return score_runtime(
                vp, st, vlm, task, nominal,
                time_stats=self.time_stats,       # ← new
                outlier_penalty=0.0,              # or whatever you like
            )
        self.vlm = vlm
        # TODO: If visual_clarity or runtime is too low, make it bad automatically
        self.criteria = {
            # "task_success":        (25, score_task_success),
            "visual_clarity":      (20, score_visual_clarity),
            "smoothness":          (10, score_smoothness),
            "collision":           (10, score_collision),
            "runtime":              (20, runtime_with_stats),
            # "path_efficiency":     (10, score_path_efficiency),
            # "joint_stability":         (5, score_joint_stability),
            # "gripper_consistency":  (5, score_gripper_consistency),
        }
        self.time_stats = time_stats
        self.norm = sum(w for w, _ in self.criteria.values())

    def score(self, video_path, states, task, nominal):
        subs, total = {}, 0.
        for k, (w, fn) in self.criteria.items():
            val = fn(video_path, states, self.vlm, task, nominal)
            subs[k] = val
            total += w * val
        return total / self.norm, subs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, type=str)
    ap.add_argument("--root", required=False, default=None, type=str)
    ap.add_argument("--output", required=False, type=str, default=None)
    ap.add_argument("--overwrite", required=False, type=bool, default=True)
    ap.add_argument("--overwrite_checkpoint", required=False, default=False)
    ap.add_argument("--nominal", type=float)
    ap.add_argument("--policy_name", type = str, default = "act")
    ap.add_argument("--threshold", type = float, default = 0.5)
    ap.add_argument("--train-baseline", type=bool, default=False)
    ap.add_argument("--train-filtered", type=bool, default=False)
    ap.add_argument("--plot", required=False, type=bool, default=False)
    args = ap.parse_args()


    # Load dataset.
    dataset = load_dataset_hf(args.repo_id, root=args.root)
    task = dataset.meta.tasks

    # This maps episode_id to video path (by camera key), states and actions.
    episode_map = organize_by_episode(dataset)


    # Compute runtimes stats of all episodes.
    states = [episode_map[i]['states'] for i in episode_map]
    time_stats = build_time_stats(states)         # ← q1, q3, mean, std, …

    scorer = DatasetScorer(None, time_stats=time_stats)#VLMInterface())
    # ------------------------------------------------------------------
    #  Evaluate every episode
    # ------------------------------------------------------------------
    rows, agg_mean = [], 0.0

    for episode_index in episode_map:
        episode = episode_map[episode_index]
        episode_total = 0
        for camera_type in episode['vid_paths']:
            vid_path = episode['vid_paths'][camera_type]
            states = episode['states']
            total, subs = scorer.score(vid_path, states, task, args.nominal)
            rows.append((episode_index, camera_type, vid_path, total, subs))
            episode_total += total
        agg_mean += episode_total / len(episode['vid_paths'])
    agg_mean /= len(rows)

    # ------------------------------------------------------------------
    #  Pretty-print results
    # ------------------------------------------------------------------
    crit_names = list(scorer.criteria.keys())

    EP_W, CAM_W, SC_W, AG_W = 8, 30, 11, 10          # col widths
    score_fmt = f'{{:>{SC_W}.3f}}'                    # one per-criterion cell
    col_fmt   = (
        f'{{:<{EP_W}}}{{:<{CAM_W}}}'                 # Episode | Camera
        + score_fmt * len(crit_names)                # each criterion
        + f'  {{:>{AG_W}.3f}}  {{}}'                 # Aggregate | Status
    )
    
    header_line = (
        f'{"Episode":<{EP_W}}{"Camera":<{CAM_W}}'
        + ''.join(f'{h:>{SC_W}s}' for h in crit_names)
        + f'  {"Aggregate":>{AG_W}s}  Status'
    )
    
    divider = '─' * len(header_line)
    
    print('\nEpisode scores (0–1 scale)')
    print(divider)
    print(header_line)

    distributions = {}
    good_episodes = {}
    for ep_idx, cam, _vid_path, total, subs in rows:
        for k in crit_names:
            if k not in distributions:
                distributions[k] = []
            distributions[k].append(subs[k])
        print(
            col_fmt.format(
                ep_idx,
                cam,
                *[subs[k] for k in crit_names],
                total,
                'GOOD' if total >= args.threshold else 'BAD'
            )
        )
        if ep_idx not in good_episodes or good_episodes[ep_idx]:
            # Check both cameras, if at least one is false, we will set it to false
            good_episodes[ep_idx] = (total >= args.threshold)

    print(divider)
    print(f'Average aggregate over {len(rows)} videos: {agg_mean:.3f}')
    print('')

    if args.output:
        good_episodes_list = [k for k in good_episodes if good_episodes[k]]
        if len(good_episodes_list) == 0:
            raise ValueError(f'All episodes filtered out, decrease threshold to fix this. Current threshold: {args.threshold}')
        total_episodes = len(episode_map)
        num_removed = total_episodes - len(good_episodes_list)

        print(f'Percentage of episodes removed: {float(num_removed)}/{total_episodes}, total: {num_removed}')
        print('')
        # Need to find actual dataset path on disk.
        dataset_path = args.root
        if not dataset_path:
            cache_dir = cache_dir = HF_LEROBOT_HOME
            dataset_path = os.path.join(cache_dir, args.repo_id)
        save_filtered_dataset(dataset_path, args.output, good_episodes_list, overwrite=args.overwrite)
        ds = load_dataset_hf(args.repo_id, root=args.output)

    # Training config required args.
    #  --dataset.repo_id=${HF_USER}/trossen_ai_stationary_test \
    #  --policy.type=act \
    #  --output_dir=outputs/train/act_trossen_ai_stationary_test \
    #  --job_name=act_trossen_ai_stationary_test \
    #  --device=cuda \
    #  --wandb.enable=true
    
    if args.train_baseline:
        if args.overwrite_checkpoints:
            print('WARNING: overwrite_checkpoints is True')
        start_training(args.repo_id, root=args.root, job_name='baseline', overwrite_checkpoint=args.overwrite_checkpoint)
    if args.train_filtered and num_removed == 0:
        print('WARNING: Not training because nothing was removed.')
    elif args.train_filtered:
        if args.overwrite_checkpoints:
            print('WARNING: overwrite_checkpoints is True')
        start_training(args.repo_id, root=args.output, job_name='filtered', overwrite_checkpoint=args.overwrite_checkpoint))

    if args.plot:
        for k in crit_names:
            uniplot.histogram(distributions[k],
                          bins=20,
                          bins_min=0,  # avoid breaking if all data lands in 1 bucket
                          title=f'distribution for {k}',
                          x_min=0,
                          x_max=1)

if __name__ == "__main__":
    main()
