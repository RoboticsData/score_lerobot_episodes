from lerobot.scripts import train as lerobot_train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
#from lerobot.envs.factory import make_env_config
from lerobot.common.policies.factory import make_policy_config
import os
from pathlib import Path


def start_training(repo_id, output_dir='./checkpoints/baseline', root=None, policy_name='act', **kwargs):
    dataset = DatasetConfig(repo_id=repo_id, root=root)
    policy = make_policy_config(policy_name)
    policy.chunk_size = 1
    policy.n_action_steps = 1

    device = 'mps'
    output_dir = Path(output_dir)
    job_name = 'act_baseline_'+repo_id.replace('/','_')
    wandb_config = WandBConfig(enable=True)

    train_config = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        wandb=wandb_config,
        output_dir=output_dir,
        job_name=job_name,
        batch_size=1,
        steps=100,
        #resume=True,
        num_workers=1)

    lerobot_train.train(train_config)
