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

### Installation
```
git clone git@github.com:RoboticsData/score_lerobot_episodes.git
cd score_lerobot_episodes

uv venv
source .venv/bin/activate
uv sync  # installs dependencies expressed in pyproject.toml and uv.lock

python -c 'import score_dataset' || echo 'Something is wrong, check dependencies'

# Test if you're logged into the hugginface CLI:
hf auth whoami
# If not, go to https://huggingface.co/settings/tokens, make a new token, and login:
hf auth login
```

### Usage

```
# in the score_lerobot_episodes directory:

source .venv/bin/activate

# Fetch a dataset from HuggingFace, example:
PATH_TO_HF_DATASET=`hf download Rorschach4153/so101_60_new --repo-type dataset`

## TODO make tool injest HF repo names instead of local paths to cached snapshots
python score_dataset.py --dataset ${PATH_TO_HF_DATASET} --task "none" --camera 'phone'
```

### Adding dependencies

Use ```uv add``` to add dependencies, then ensure you commit changes to pyproject.toml and uv.lock.

#### Python versions

We opt for keeping ```.python-version``` in the repo as a way to interlock the Python
runtime version with installed dependecy versions. Python packages express
dependence on difference Python versions, so Python itself effectively becomes
a dependency.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RoboticsData/score_lerobot_episodes&type=Date)](https://www.star-history.com/#RoboticsData/score_lerobot_episodes&Date)

## Term of Use
LeRobot Episode Scoring Toolkit is distributed under the Apache 2.0 license.
