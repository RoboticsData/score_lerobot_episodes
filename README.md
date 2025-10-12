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
| Gripper consistency         | `score_gripper_consistency`                         | Binary "closed vs. holding" agreement                        |
| Actuator saturation         | `score_actuator_saturation`                         | Difference between commanded actions and achieved states     |
| Task success (VLM)          | `score_task_success` (via `VLMInterface`)           | Gemini grades whether the desired behaviour happened         |
| Runtime penalty / outliers  | `score_runtime` + `build_time_stats`, `is_time_outlier` | Episode length vs. nominal / Tukey-IQR / Z-score fences      |

---

## ⚙️  Installation

```bash
# clone your repo first
pip install -r requirements.txt
export GOOGLE_API_KEY="sk-..."      # Required only if you use VLM-based scoring
```
Note: The free tier rate limits of the Gemini API, are fairly restrictive and might need to be upgraded depending on how long the episodes are. Check https://ai.google.dev/gemini-api/docs/rate-limits for more info.

## Example
```bash
python score_dataset.py --repo_id Daddyboss/so_100_test4 --output ./output/Daddyboss/so_100_test4 --train-baseline=True --train-filtered=True
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RoboticsData/score_lerobot_episodes&type=Date)](https://www.star-history.com/#RoboticsData/score_lerobot_episodes&Date)

## Term of Use
LeRobot Episode Scoring Toolkit is distributed under the Apache 2.0 license.
