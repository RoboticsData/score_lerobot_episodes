from data import load_dataset_hf
from contextlib import nullcontext
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from torch.utils.data import DataLoader, default_collate
import torch
import wandb

def get_eval_episodes(good_episodes, eval_percentage=1.0):
    num_episodes = int(eval_percentage * len(good_episodes))
    # For baseline, good_episodes is the eval
    baseline_eval_episodes = good_episodes[:num_episodes]
    # For filtered, last few episodes are eval
    filtered_eval_episodes = list(range(num_episodes))
    return baseline_eval_episodes, filtered_eval_episodes

def move_to_device(data, device):
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def run_eval(policy_path, repo_id, wandb_id, episodes, use_amp=False, root=None):
    wandb.init(project="lerobot", id=wandb_id, resume="must")
    dataset = load_dataset_hf(repo_id, episodes=episodes)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        #collate_fn=default_collate,  # works if each item is a dict of tensors
        drop_last=False,
        pin_memory=False,            # True if device is cuda
    )

    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    policy = make_policy(policy_config, dataset.meta)
    if hasattr(policy.config, 'use_vae'):
        # Special case for ACT.
        policy.config.use_vae = False
    policy.training = False
    policy.eval()
    total_loss = 0
    with torch.no_grad(), torch.autocast(device_type=device.type) if use_amp else nullcontext():
        for i, batch in enumerate(loader):
            batch['action_is_pad'] = torch.zeros(policy.config.n_action_steps).bool()
            print(f'Evaluating batch: {i}/{len(loader)}')
            batch = move_to_device(batch, policy.config.device)
            loss, output_dict = policy.forward(batch)
            total_loss += loss
    mean_loss = total_loss / len(loader)
    print(f'Mean loss: {mean_loss}')
    # TODO: Add wandb logging
    wandb.log({"eval/mean_loss": mean_loss})
    wandb.finish()
    return total_loss
