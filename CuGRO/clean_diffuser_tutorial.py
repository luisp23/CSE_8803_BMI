
import os
import gym
import d4rl
import numpy as np
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d

from cleandiffuser.nn_condition import BaseNNCondition, get_mask

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.utils import loop_dataloader




class ValueNNCondition(BaseNNCondition):
    """ Simple MLP NNCondition for value conditioning.
    
    value (bs, 1) -> ValueNNCondition -> embedding (bs, emb_dim)

    Args:
        emb_dim (int): Embedding dimension.
        dropout (float): Label dropout rate.
    
    Example:
        >>> value = torch.rand(32, 1)
        >>> condition = ValueNNCondition(emb_dim=64, dropout=0.25)
        >>> # If condition.training, embedding will be masked to be dummy condition 
        >>> # with label dropout rate 0.25.
        >>> embedding = condition(value) 
        >>> embedding.shape
        torch.Size([32, 64])
    """
    def __init__(self, emb_dim: int, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(1, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, emb_dim))
    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = get_mask(
            mask, (condition.shape[0],), self.dropout, self.training, condition.device)
        mask = at_least_ndim(mask, condition.dim())
        return condition * mask




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training with Diffusion Models")
    
    # Add a single string argument
    parser.add_argument("--mode", type=str, default="training", 
                        help="mode (training/evaluation)")
    
    # Parse arguments
    args = parser.parse_args()


    # horizon=4 is enough for halfcheetah tasks as mentioned in Diffuser paper.
    horizon = 4
    env = gym.make("halfcheetah-medium-expert-v2")
    
    
    
    # TODO: this makes datset with specified horizon, how to do we adapt to dataset in the CuGRO? do we need to?
    # will we need plug in this dataset class into CuGRO training to get right conditioning data (e.g. trajectory/horizon)
    # this will need to be done; 
    # look at: CleanDiffuser/cleandiffuser/dataset/d4rl_mujoco_dataset.py  
    # and: continual_diffuser/CuGRO/dataset.py 
    
    # takes as input dataset["observations"], etc.. 
    # change CuGRO to use wrapper D4RLMuJoCoDataset, take in what CuGRO does in their _load_data function;
    # process outputs for training as the D4RLMuJoCoDataset does
    
    dataset = D4RLMuJoCoDataset(env.get_dataset(), terminal_penalty=-100, horizon=horizon)
    
    
    
    
    
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"



    # TODO: need to add inverse dynamics model part
    # CleanDiffuser/pipelines/dd_d4rl_mujoco.py   or pipelines/dd_d4rl_kitchen.py or  pipelines/dd_d4rl_antmaze.py for examples
    
    # see setup for Unet here: CleanDiffuser/pipelines/dd_d4rl_mujoco.py

    # params from CleanDiffuser/configs/diffuser/mujoco/task/halfcheetah-medium-expert-v2.yaml
    
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, dim_mult=[1, 4, 2],
        timestep_emb_type="positional", attention=False, kernel_size=5)
    
    

    # alternatively could just use DiT1d  as network 
    # nn_diffusion = DiT1d(
    #     obs_dim + act_dim, emb_dim=128, d_model=320, n_heads=10, depth=2, 
    #     timestep_emb_type="untrainable_fourier")
    
    
    
    
    
    
    
    
    nn_condition = ValueNNCondition(emb_dim=128, dropout=0.25)

    # this basically says keep the observation at horizon timestep 0
    fix_mask = torch.zeros((horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    
    loss_weight = torch.ones((horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = 10.

    planner = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.9999,
        device=device)



    random_obs = torch.randn((obs_dim,))
    
    

    prior = torch.zeros((1, horizon, obs_dim + act_dim))
        
    # condition "prior" on the observation in the horizon at t=0
    prior[:, 0, :obs_dim] = random_obs[None, :]

    
    # traj has filled  in everything thats not 1 in the fixed mask; "inpainting of trajectory"
    traj, log = planner.sample(
        prior, solver="ddpm", n_samples=1, sample_steps=5)

    print(f'Trajectory shape: {traj.shape}')
    print(f'First observation MSE: {(traj[0, 0, :obs_dim].cpu() - random_obs).pow(2).mean()}')
    

    savepath = "../tutorials/results/2_classifier_free_guidance/"
    if args.mode == "training":
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        dataloader = DataLoader(
            dataset, batch_size=64, shuffle=True, num_workers=1, persistent_workers=True)

        n_gradient_steps = 0
        avg_loss = 0.
        planner.train()
        
        for batch in loop_dataloader(dataloader):
            start_time = time.time()  
            
            # bs: batch_size, h: horizon 
            
            # obs/act: (bs, h, obs_dim/act_dim)
            # these needs to be the full horizon observation in the trajectory 
            # the masking and training part is taken care of by the planner object 
            
            obs, act = batch["obs"]["state"].to(device), batch["act"].to(device)
            
            # val (bs, 1)
            # this is how its being conditioned on reward
            val = batch["val"].to(device) / 1200.  # (normalize such that 1.0 is the highest value)
            x0 = torch.cat([obs, act], dim=-1)

            avg_loss += planner.update(x0=x0, condition=val)["loss"]
            
            n_gradient_steps += 1
            iteration_time = time.time() - start_time  
            if n_gradient_steps % 1000 == 0:
                print(f'Step: {n_gradient_steps} | Loss: {avg_loss / 1000} | Time/Iter: {iteration_time:.4f} sec')
                avg_loss = 0.
            
            if n_gradient_steps % 10_000 == 0:
                print(f"saving to {savepath}/diffusion.pt")
                planner.save(savepath + "diffusion.pt")
            
            if n_gradient_steps == 500_000:
                break
    
    
    if args.mode == "evaluation":
        solver = "ddpm"
        sampling_step = 5
        num_episodes = 3
        num_envs = 50
        target_return = 0.95
        w_cfg = 1.2

        planner.load(savepath + "diffusion.pt")
        planner.eval()

        # Parallelize evaluation
        env_eval = gym.vector.make('halfcheetah-medium-expert-v2', num_envs=num_envs)

        # Get normalizers
        normalizer = dataset.get_normalizer()

        episode_rewards = []

        prior = torch.zeros((num_envs, horizon, obs_dim + act_dim), device=device)
        condition = torch.ones((num_envs, 1), device=device) * target_return
        for i in range(num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)

                # sample trajectories
                prior[:, 0, :obs_dim] = obs
                traj, log = planner.sample(
                    prior, 
                    solver=solver,
                    n_samples=num_envs, 
                    sample_step_schedule="quad_continuous",
                    sample_steps=sampling_step, use_ema=True,
                    condition_cfg=condition, w_cfg=w_cfg, temperature=1.0)
                act = traj[:, 0, obs_dim:].clip(-1., 1.).cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        mean_rewards = np.mean(episode_rewards, -1) * 100.
        print(f'D4RL score: {mean_rewards.mean():.3f} +- {mean_rewards.std():.3f}')
        env_eval.close()



