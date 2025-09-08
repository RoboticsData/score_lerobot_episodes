from data import load_dataset_hf
from contextlib import nullcontext
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader, default_collate
import torch

def get_eval_episodes(good_episodes, eval_percentage=1.0):
    num_episodes = int(eval_percentage * len(good_episodes))
    # For baseline, good_episodes is the eval
    baseline_eval_episodes = good_episodes[:num_episodes]
    # For filtered, last few episodes are eval
    filtered_eval_episodes = list(range(num_episodes))
    return baseline_eval_episodes, filtered_eval_episodes

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

    policy = ACTPolicy.from_pretrained(policy_path)
    policy = policy.to('cpu')
    policy.config.use_vae = False
    policy.training = False
    policy.eval()
    total_loss = 0
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        for i, batch in enumerate(loader):
            batch['action_is_pad'] = torch.zeros(policy.config.n_action_steps).bool()
            #print(batch.keys())
            print(f'Evaluating batch: {i}/{len(loader)}')
            #batch = batch.to('mps')
            loss, output_dict = policy.forward(batch)
            total_loss += loss
    print(f'Mean loss: {total_loss/len(loader)}')
    # TODO: Add wandb logging
    wandb.log({"eval/mean_loss": mean_loss})
    wandb.finish()
    return total_loss
