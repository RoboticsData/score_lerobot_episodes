<p align="center">
  <img src="https://raw.githubusercontent.com/sammyatman/score_lerobot_episodes/refs/heads/main/LeRobotEpisodeScoringToolkit.png" height="350" alt="LeRobotEpisodeScoringToolkit" />
</p>
<p align="center">
  <em>A lightweight toolkit for quantitatively scoring LeRobot episodes.</em>
</p>
<p align="center">

# **LeRobot Episode Scoring Toolkit**

It combines classic Computer Vision heuristics (blur / exposure tests, kinematic smoothness, collision spikes …) with optional Gemini-powered vision–language checks to give each episode a **0 – 1 score** for multiple quality dimensions.

---

## ✨  Features
| Dimension                   | Function                                            | What it measures                                             |
| --------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Visual clarity              | `score_visual_clarity`                              | Blur, over/under-exposure, low-light frames                  |
| Smoothness                  | `score_smoothness`                                  | 2-nd derivative of joint angles                              |
| Path efficiency             | `score_path_efficiency`                             | Ratio of straight-line vs. actual joint-space path           |
| Collision / spikes          | `score_collision`                                   | Sudden acceleration outliers (proxy for contacts)            |
| Joint stability (final 2 s) | `score_joint_stability`                             | Stillness at the goal pose                                   |
| Gripper consistency         | `score_gripper_consistency`                         | Binary “closed vs. holding” agreement                        |
| Task success (VLM)          | `score_task_success` (via `VLMInterface`)           | Gemini grades whether the desired behaviour happened         |
| Runtime penalty / outliers  | `score_runtime` + `build_time_stats`, `is_time_outlier` | Episode length vs. nominal / Tukey-IQR / Z-score fences      |

---

## ⚙️  Installation

```bash
# clone your repo first
pip install -r requirements.txt
export GOOGLE_API_KEY="sk-..."      # Required only if you use VLM-based scoring

## Example
HF_USER='...'
python score_dataset.py --dataset /path/to/data/${HF_USER}/open-book --task "Open the book"
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RoboticsData/score_lerobot_episodes&type=Date)](https://www.star-history.com/#RoboticsData/score_lerobot_episodes&Date)

## Term of Use
LeRobot Episode Scoring Toolkit is distributed under the Apache 2.0 license.
