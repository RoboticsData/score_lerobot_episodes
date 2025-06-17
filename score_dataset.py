import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple

from vlm import VLMInterface
from data import discover_episodes, load_state_from_parquet
from scores import score_task_success, score_visual_clarity, score_smoothness, score_path_efficiency, score_collision, score_runtime, score_joint_stability, score_gripper_consistency


class DatasetScorer:
    def __init__(self, vlm: VLMInterface):
        self.vlm = vlm
        self.criteria = {
            "task_success":        (25, score_task_success),
            "visual_clarity":      (10, score_visual_clarity),
            "smoothness":          (15, score_smoothness),
            "path_efficiency":     (10, score_path_efficiency),
            "collision":           (15, score_collision),
            "runtime":              (5, score_runtime),
            "joint_stability":         (5, score_joint_stability),
            "gripper_consistency":  (5, score_gripper_consistency),
        }
        self.norm = sum(w for w, _ in self.criteria.values())

    def score(self, video_path, state, task, nominal):
        subs, total = {}, 0.
        for k, (w, fn) in self.criteria.items():
            val = fn(video_path, state, self.vlm, task, nominal)
            subs[k] = val
            total += w * val
        return total / self.norm, subs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=pathlib.Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--nominal", type=float)
    ap.add_argument("--camera", default="shoulder", choices=["shoulder", "wrist"])
    args = ap.parse_args()

    pairs = discover_episodes(args.dataset, args.camera)
    scorer = DatasetScorer(None)#VLMInterface())
    rows, mean = [], 0.

    for vid_path, pq_path in pairs:
        state = load_state_from_parquet(pq_path)
        score, _ = scorer.score(vid_path, state, args.task, args.nominal)
        rows.append((vid_path.name, score))
        mean += score

    mean /= len(rows)
    print("\nEpisode scores")
    print("──────────────")
    for name, s in rows:
        print(f"{name:<20s}  {s:.3f}  {'GOOD' if s >= 0.75 else 'BAD'}")
    print("──────────────")
    print(f"Average over {len(rows)} episodes: {mean:.3f}")

if __name__ == "__main__":
    main()
