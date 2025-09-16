import argparse, sys, os
import pandas as pd
import json
from moviepy import VideoFileClip

#from data import organize_by_episode, load_dataset_hf, save_filtered_dataset
#from scores import score_task_success, score_visual_clarity, score_smoothness, score_path_efficiency, score_collision, score_runtime, score_joint_stability, score_gripper_consistency
#from scores import build_time_stats           # (your helper from the other file)
#from score_dataset import DatasetScorer
#from data import load_dataset_hf, organize_by_episode

from lerobot.constants import HF_LEROBOT_HOME

def get_pandas_df(results_path):
    if not results_path or not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")
    
    if os.path.getsize(results_path) == 0:
        raise ValueError(f"Results file is empty: {results_path}")

    with open(results_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in results file: {results_path}") from e
        
    return pd.DataFrame(data)

def get_quantiles(results_path, num_quantiles=4, keep_camera=True):
    df = get_pandas_df(results_path)

    # mean score per episode
    episode_scores = (
        df.groupby("episode_id")["aggregate_score"]
        .mean()
        .reset_index()
    )
    episode_scores.rename(columns={"aggregate_score": "mean_score"}, inplace=True)

    # assign quantiles
    quantiles, bin_edges = pd.qcut(
        episode_scores["mean_score"],
        q=num_quantiles,
        labels=False,
        retbins=True
    )
    episode_scores["quantile"] = quantiles

    if keep_camera:
        # merge quantile info back into the full dataframe (preserves video_path!)
        df_with_quantiles = df.merge(
            episode_scores[["episode_id", "quantile"]],
            on="episode_id"
        )
    else:
        df_with_quantiles = episode_scores

    return df_with_quantiles, bin_edges


def visualize_quantile(df, quantile_pick=0, n_samples=3):
    lowest = df[df["quantile"] == quantile_pick]
    
    # pick unique episodes
    sampled_eps = lowest["episode_id"].drop_duplicates().sample(n=n_samples, random_state=42)
    
    for ep in sampled_eps:
        vid_path = lowest[lowest["episode_id"] == ep]["video_path"].iloc[0]
        print(f"Episode {ep}, video: {vid_path}")
        
        clip = VideoFileClip(vid_path)  # full clip
        temp_file = f"episode_{ep:06d}.mp4"
        clip.write_videofile(temp_file, codec="libx264")
        clip.close()
        os.system(f"open {temp_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, type=str)
    ap.add_argument("--results_path", required=False, default=None, type=str)
    ap.add_argument("--quantiles", type = int, default = 4)
    ap.add_argument("--quantile_pick", type = int, default = 0)
    ap.add_argument("--num_samples", type = int, default = 3)
    args = ap.parse_args()

    #load JSON results
    if args.results_path:
        results_path = args.results_path
    #if results path not provided, we consider the default path given by score_dataset.py
    else:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        repo_name = args.repo_id.replace("/", "_")
        results_path = os.path.join(results_dir, f"{repo_name}_scores.json")
    
    print(f"Using results file: {results_path}")
    
    #get quantiles:
    df, quantiles = get_quantiles(results_path, num_quantiles = args.quantiles)
    #print(df.head())
    for i, edge in enumerate(quantiles):
        print(f"Quantile {i} cutoff: {edge}")
    

    #sample episodes from the picked quantile
    visualize_quantile(df, quantile_pick=args.quantile_pick, n_samples=args.num_samples)



if __name__ == "__main__":
    main()


    








