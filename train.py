from lerobot.scripts import train as lerobot_train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
#from lerobot.envs.factory import make_env_config
from lerobot.policies.factory import make_policy_config
import os
from pathlib import Path
import wandb
import shutil
import torch

def start_training(repo_id, root=None, output_dir=None, policy_name='act', job_name='', overwrite_checkpoint=False, **kwargs):
    dataset = DatasetConfig(repo_id=repo_id, root=root)
    policy = make_policy_config(policy_name)
    policy.push_to_hub = False
    #policy.chunk_size = 1
    #policy.n_action_steps = 1

    device = 'mps'

    dataset_name = repo_id.replace('/', '_')
    full_job_name = f"{job_name or 'train'}_{policy_name}_{dataset_name}"

    if not output_dir:
        output_dir = f'./checkpoints/{full_job_name}'

    #lerobot_train will give an error if resume is False in train_config and output_dir is non-empty  
    if overwrite_checkpoint and os.path.exists(output_dir):
        print(f'Removing directory: {output_dir}')
        shutil.rmtree(output_dir)
    
    
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
        #resume=True,
        num_workers=4)

    lerobot_train.train(train_config)
    wandb_id = wandb.run.id if wandb.run else None

    wandb.finish()
    return output_dir, wandb_id

if __name__ == '__main__':
    repo_id = 'sammyatman/open-book'
    root="./output/sammyatman/open-book"
    start_training(repo_id, output_dir='./checkpoints/baseline', root=root, policy_name='act')
