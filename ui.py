import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import sys
from pathlib import Path

from score_lerobot_episodes.vlm import VLMInterface
from score_lerobot_episodes.data import organize_by_episode, load_dataset_hf, save_filtered_dataset
from score_lerobot_episodes.scores import score_task_success, score_visual_clarity, score_smoothness, score_path_efficiency, score_collision, score_runtime, score_joint_stability, score_gripper_consistency
from score_lerobot_episodes.scores import build_time_stats
from score_dataset import DatasetScorer

st.set_page_config(
    page_title="LeRobot Episode Scoring Toolkit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_scoring_dashboard(results_df, distributions, agg_mean, criteria_names):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Episode Scores Overview")
        
        fig = px.scatter(
            results_df, 
            x="Episode", 
            y="Aggregate Score",
            color="Status",
            color_discrete_map={"GOOD": "#2ecc71", "BAD": "#e74c3c"},
            hover_data=["Camera"] + criteria_names,
            title="Episode Performance by Aggregate Score"
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold (0.5)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        good_episodes = len(results_df[results_df["Status"] == "GOOD"])
        total_episodes = len(results_df)
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Total Episodes", total_episodes)
        with col_metric2:
            st.metric("Good Episodes", good_episodes, f"{good_episodes/total_episodes*100:.1f}%")
        with col_metric3:
            st.metric("Average Score", f"{agg_mean:.3f}")
    
    with col2:
        st.subheader("Score Distribution")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=results_df["Aggregate Score"],
            nbinsx=20,
            marker_color="#3498db",
            opacity=0.7
        ))
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_hist.update_layout(
            title="Aggregate Score Distribution",
            xaxis_title="Score",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def create_criteria_analysis(distributions, criteria_names):
    st.subheader("Criteria Analysis")
    
    n_criteria = len(criteria_names)
    cols = st.columns(min(3, n_criteria))
    
    for i, criterion in enumerate(criteria_names):
        with cols[i % 3]:
            scores = distributions[criterion]
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=scores,
                name=criterion,
                marker_color="#9b59b6"
            ))
            fig.update_layout(
                title=f"{criterion.replace('_', ' ').title()}",
                yaxis_title="Score",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            avg_score = np.mean(scores)
            st.metric(f"Avg {criterion}", f"{avg_score:.3f}")

def create_detailed_table(results_df):
    st.subheader("Detailed Results")
    
    status_filter = st.selectbox("Filter by Status", ["All", "GOOD", "BAD"])
    
    filtered_df = results_df
    if status_filter != "All":
        filtered_df = results_df[results_df["Status"] == status_filter]
    
    st.dataframe(
        filtered_df.style.format({
            "Aggregate Score": "{:.3f}",
            **{col: "{:.3f}" for col in filtered_df.columns if col not in ["Episode", "Camera", "Video Path", "Status"]}
        }).applymap(
            lambda x: "background-color: #d5f4e6" if x == "GOOD" else "background-color: #ffeaa7" if x == "BAD" else "",
            subset=["Status"]
        ),
        use_container_width=True
    )

def run_scoring_analysis(repo_id, root_path, nominal_time):
    with st.spinner("Loading dataset and analyzing episodes..."):
        try:
            dataset = load_dataset_hf(repo_id, root=root_path)
            task = dataset.meta.tasks
            episode_map = organize_by_episode(dataset)
            
            states = [episode_map[i]['states'] for i in episode_map]
            time_stats = build_time_stats(states)
            
            scorer = DatasetScorer(None, time_stats=time_stats)
            
            rows = []
            agg_mean = 0.0
            
            progress_bar = st.progress(0)
            total_episodes = len(episode_map)
            
            for idx, episode_index in enumerate(episode_map):
                episode = episode_map[episode_index]
                episode_total = 0
                
                for camera_type in episode['vid_paths']:
                    vid_path = episode['vid_paths'][camera_type]
                    states = episode['states']
                    total, subs = scorer.score(vid_path, states, task, nominal_time)
                    rows.append((episode_index, camera_type, vid_path, total, subs))
                    episode_total += total
                
                agg_mean += episode_total / len(episode['vid_paths'])
                progress_bar.progress((idx + 1) / total_episodes)
            
            agg_mean /= len(rows)
            
            criteria_names = list(scorer.criteria.keys())
            distributions = {k: [] for k in criteria_names}
            
            results_data = []
            for ep_idx, cam, vid_path, total, subs in rows:
                row_data = {
                    "Episode": ep_idx,
                    "Camera": cam,
                    "Video Path": vid_path,
                    "Aggregate Score": total,
                    "Status": "GOOD" if total >= 0.5 else "BAD"
                }
                
                for k in criteria_names:
                    distributions[k].append(subs[k])
                    row_data[k] = subs[k]
                
                results_data.append(row_data)
            
            results_df = pd.DataFrame(results_data)
            
            return results_df, distributions, agg_mean, criteria_names, episode_map
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return None, None, None, None, None

def main():
    st.title("🤖 LeRobot Episode Scoring Toolkit")
    st.markdown("Analyze and visualize robot episode performance with interactive dashboards")
    
    with st.sidebar:
        st.header("Configuration")
        
        repo_id = st.text_input(
            "Repository ID",
            placeholder="e.g., lerobot/svla_so101_pickplace",
            help="HuggingFace repository ID for the dataset"
        )
        
        root_path = st.text_input(
            "Root Path (Optional)",
            placeholder="Leave empty for default cache",
            help="Local path to dataset root directory"
        )
        
        nominal_time = st.number_input(
            "Nominal Time",
            min_value=0.0,
            value=10.0,
            step=0.1,
            help="Reference time for runtime scoring"
        )
        
        analyze_button = st.button("🔍 Analyze Episodes", type="primary")
        
        st.markdown("---")
        
        with st.expander("Export Options"):
            export_filtered = st.checkbox("Save filtered dataset")
            output_path = st.text_input("Output Path", value="./filtered_output")
            
        with st.expander("Training Options"):
            train_baseline = st.checkbox("Train baseline model")
            train_filtered = st.checkbox("Train filtered model")
    
    if analyze_button and repo_id:
        root = root_path if root_path.strip() else None
        
        results_df, distributions, agg_mean, criteria_names, episode_map = run_scoring_analysis(
            repo_id, root, nominal_time
        )
        
        if results_df is not None:
            st.session_state.results_df = results_df
            st.session_state.distributions = distributions
            st.session_state.agg_mean = agg_mean
            st.session_state.criteria_names = criteria_names
            st.session_state.episode_map = episode_map
            st.session_state.repo_id = repo_id
            st.session_state.root_path = root
            st.session_state.output_path = output_path
            
            st.success(f"✅ Analysis complete! Processed {len(results_df)} video segments from {len(results_df['Episode'].unique())} episodes.")
    
    if hasattr(st.session_state, 'results_df'):
        create_scoring_dashboard(
            st.session_state.results_df,
            st.session_state.distributions,
            st.session_state.agg_mean,
            st.session_state.criteria_names
        )
        
        st.markdown("---")
        
        create_criteria_analysis(
            st.session_state.distributions,
            st.session_state.criteria_names
        )
        
        st.markdown("---")
        
        create_detailed_table(st.session_state.results_df)
        
        with st.sidebar:
            if export_filtered and st.button("💾 Export Filtered Dataset"):
                with st.spinner("Exporting filtered dataset..."):
                    try:
                        good_episodes = st.session_state.results_df[
                            st.session_state.results_df["Status"] == "GOOD"
                        ]["Episode"].unique().tolist()
                        
                        dataset_path = st.session_state.root_path
                        if not dataset_path:
                            cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot/")
                            dataset_path = os.path.join(cache_dir, st.session_state.repo_id)
                        
                        save_filtered_dataset(dataset_path, st.session_state.output_path, good_episodes)
                        st.success(f"✅ Filtered dataset saved to {st.session_state.output_path}")
                    except Exception as e:
                        st.error(f"Error exporting dataset: {str(e)}")
    
    if not hasattr(st.session_state, 'results_df'):
        st.info("👆 Enter a repository ID and click 'Analyze Episodes' to get started!")
        
        with st.expander("ℹ️ How to use this tool"):
            st.markdown("""
            1. **Repository ID**: Enter a HuggingFace repository ID (e.g., `lerobot/svla_so101_pickplace`)
            2. **Root Path**: Optional local path to dataset (leave empty to use HF cache)
            3. **Nominal Time**: Reference time for runtime scoring (usually episode duration)
            4. **Click Analyze**: Process episodes and generate interactive visualizations
            
            The tool will evaluate each episode across multiple criteria:
            - **Visual Clarity**: Image quality and sharpness
            - **Smoothness**: Motion smoothness and consistency
            - **Collision**: Collision detection and avoidance
            - **Runtime**: Execution time efficiency
            """)

if __name__ == "__main__":
    main()
