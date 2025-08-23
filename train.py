from lerobot.scripts import train as lerobot_train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
#from lerobot.envs.factory import make_env_config
from lerobot.common.policies.factory import make_policy_config
import os
from pathlib import Path
import wandb
import torch

def start_training(repo_id, root=None, output_dir=None, policy_name='act', job_name='', **kwargs):
    dataset = DatasetConfig(repo_id=repo_id, root=root)
    policy = make_policy_config(policy_name)
    #policy.chunk_size = 1
    #policy.n_action_steps = 1

    device = 'mps'

    dataset_name = repo_id.replace('/', '_')
    full_job_name = f"{job_name or 'train'}_{policy_name}_{dataset_name}"

    if not output_dir:
        output_dir = f'./checkpoints/{full_job_name}'
    
    output_dir = Path(output_dir)
    wandb_config = WandBConfig(enable=True)

    train_config = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        wandb=wandb_config,
        output_dir=output_dir,
        job_name=full_job_name,
        batch_size=4,
        steps=10000,
        log_freq=200,
        eval_freq=200,
        #eval_holdout_split="0:5",
        #resume=True,
        num_workers=4)

    lerobot_train.train(train_config)
    wandb.finish()

if __name__ == '__main__':
    repo_id = 'sammyatman/open-book'
    root="./output/sammyatman/open-book"
    start_training(repo_id, output_dir='./checkpoints/baseline', root=root, policy_name='act')
