import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
from typing import Dict, List, Tuple

from vlm import VLMInterface
from data import discover_episodes, load_state_from_parquet
from scores import score_task_success, score_visual_clarity, score_smoothness, score_path_efficiency, score_collision, score_runtime, score_joint_stability, score_gripper_consistency
from scores import build_time_stats           # (your helper from the other file)


class DatasetScorer:
    def __init__(self, vlm: VLMInterface, time_stats: dict):
        def runtime_with_stats(vp, st, vlm, task, nominal):
            return score_runtime(
                vp, st, vlm, task, nominal,
                time_stats=self.time_stats,       # ← new
                outlier_penalty=0.0,              # or whatever you like
            )
        self.vlm = vlm
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
    states = [load_state_from_parquet(pq) for _, pq in pairs]
    time_stats = build_time_stats(states)         # ← q1, q3, mean, std, …
    scorer = DatasetScorer(None, time_stats=time_stats)#VLMInterface())
    # ------------------------------------------------------------------
    #  Evaluate every episode
    # ------------------------------------------------------------------
    rows, agg_mean = [], 0.0

    for vid_path, pq_path in pairs:
        state = load_state_from_parquet(pq_path)
        total, subs = scorer.score(vid_path, state, args.task, args.nominal)
        rows.append((vid_path.name, total, subs))
        agg_mean += total

    agg_mean /= len(rows)

    # ------------------------------------------------------------------
    #  Pretty-print results
    # ------------------------------------------------------------------
    crit_names = list(scorer.criteria.keys())         # preserve order
    header = ["Episode"] + crit_names + ["Aggregate", "Status"]
    col_fmt = "{:<20}" + "{:>11.3f}" * len(crit_names) + "  {:>10.3f}  {}"

    print("\nEpisode scores (0 – 1 scale)")
    print("─" * (20 + 12 * (len(crit_names) + 2)))

    # header row
    print("{:<20}".format(header[0]) +
          "".join(f"{h:>11s}" for h in header[1:-1]) +
          f"  {header[-2]:>10s}  {header[-1]}")

    # episode rows
    for name, total, subs in rows:
        status = "GOOD" if total >= 0.5 else "BAD"
        per_cat = [subs[k] for k in crit_names]
        print(col_fmt.format(name, *per_cat, total, status))

    print("─" * (20 + 12 * (len(crit_names) + 2)))
    print(f"Average aggregate over {len(rows)} episodes: {agg_mean:.3f}")


if __name__ == "__main__":
    main()
