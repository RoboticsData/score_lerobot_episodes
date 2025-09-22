import argparse, sys, os
import pandas as pd
import json
from moviepy import VideoFileClip

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
    df = pd.DataFrame(data)
    df_expanded = pd.concat([df.drop(columns= ["per_attribute_scores"]), df["per_attribute_scores"].apply(pd.Series)], axis = 1)
    return df_expanded

def get_quantiles(results_path, num_quantiles=4, keep_camera=True, col = "aggregate_score"):
    df = get_pandas_df(results_path)

    # mean score per episode
    episode_scores = (
        df.groupby("episode_id")[col]
        .mean()
        .reset_index()
    )
    episode_scores.rename(columns={col: "mean_score"}, inplace=True)

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


def visualize_quantile(df, quantile_pick=0, n_samples=3, save_path = None):
    lowest = df[df["quantile"] == quantile_pick]
    
    # pick unique episodes
    sampled_eps = lowest["episode_id"].sample(n=n_samples, random_state=42)

    for ep in sampled_eps:
        episode_rows = lowest[lowest["episode_id"] == ep]

        for _, row in episode_rows.iterrows():
            vid_path = row["video_path"]
            cam_type = row["camera_type"]
            print(f"Episode {ep}, camera: {cam_type}, video: {vid_path}")

            try:
                clip = VideoFileClip(vid_path)  # full clip
                filename = f"episode_{ep:06d}_{cam_type}.mp4"

                if save_path:
                    out_file = os.path.join(save_path, filename)
                else:
                    out_file = filename
                
                #Save the video
                clip.write_videofile(out_file, codec="libx264")
                print(f"  Saved to: {out_file}")

                #Open the video
                os.system(f"open '{out_file}'")
                clip.close()

            #raise Exception if video file not found
            except Exception as e:
                print(f"  Error processing {vid_path}: {str(e)}")
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, type=str)
    ap.add_argument("--results_path", required=False, default=None, type=str)
    ap.add_argument("--save_path", required=False, default=None, type=str)
    ap.add_argument("--quantiles", type = int, default = 4)
    ap.add_argument("--quantile_pick", type = int, default = 0)
    ap.add_argument("--num_samples", type = int, default = 3)
    ap.add_argument("--type", required=False, choices=["aggregate_score", "visual_clarity", "smoothness", "collision"], default="aggregate_score")
    ap.add_argument("--csv", type=bool, default=False)
    args = ap.parse_args()

    #set repo_name
    repo_name = args.repo_id.replace("/", "_")

    #load JSON results
    if args.results_path:
        results_path = args.results_path
        

    #if results path not provided, we consider the default path given by score_dataset.py
    else:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{repo_name}_scores.json")
    
    print(f"Using results file: {results_path}")
    
    #get quantiles:
    df, quantiles = get_quantiles(results_path, num_quantiles = args.quantiles, col = args.type)

    if args.csv:
        csv_path = f'{repo_name}_quantiles.csv'
        df.to_csv(csv_path, index=False)

    for i, edge in enumerate(quantiles):
        print(f"Quantile {i} cutoff: {edge}")
    
    if args.save_path:
        save_path = args.save_path
    else:
        save_dir = "saved_examples"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{repo_name}_examples")
        os.makedirs(save_path, exist_ok=True)
    
    #sample episodes from the picked quantile
    visualize_quantile(df, quantile_pick=args.quantile_pick, n_samples=args.num_samples, save_path = save_path)


if __name__ == "__main__":
    main()


    








