from lerobot.scripts import train as lerobot_train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
#from lerobot.envs.factory import make_env_config
from lerobot.common.policies.factory import make_policy_config
import os
from pathlib import Path
import wandb

def start_training(repo_id, root=None, output_dir='./checkpoints/baseline', policy_name='act', job_name='', **kwargs):
    dataset = DatasetConfig(repo_id=repo_id, root=root)
    policy = make_policy_config(policy_name)
    #policy.chunk_size = 1
    #policy.n_action_steps = 1

    device = 'mps'
    output_dir = Path(output_dir)
    dataset_name = repo_id.replace('/', '_')
    full_job_name = f"{job_name or 'train'}_{policy_name}_{dataset_name}"
    wandb_config = WandBConfig(enable=True)

    train_config = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        wandb=wandb_config,
        output_dir=output_dir,
        job_name=full_job_name,
        batch_size=8,
        steps=100,
        log_freq=1,
        #resume=True,
        num_workers=0)

    lerobot_train.train(train_config)
    wandb.finish()

if __name__ == '__main__':
    repo_id = 'sammyatman/open-book'
    root="./output/sammyatman/open-book"
    start_training(repo_id, output_dir='./checkpoints/baseline', root=root, policy_name='act')
