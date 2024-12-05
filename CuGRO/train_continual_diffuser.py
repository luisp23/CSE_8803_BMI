import os
from tqdm import tqdm, trange
import functools
import time
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, MlpScoreNet, GenerateNet, MlpGenerateNet
from utils import get_args
from dataset import Diffusion_buffer
import time
import json
from src.envs import  HalfCheetahVelEnv, WalkerRandParamsWrappedEnv, SwimmerDir
from collections import namedtuple
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP


import yaml
import torch.distributed as dist
from continualworld.envs import get_cl_env
from continualworld.tasks import TASK_SEQS
from normalization import DatasetNormalizer
from typing import Union, Dict, Callable



from torch.utils.data import DataLoader

import torch.nn as nn

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import DiT1d, JannerUNet1d

from cleandiffuser.nn_condition import BaseNNCondition, get_mask

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.utils import loop_dataloader


def Tasks(config):
    with open(config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    tasks = []
    for task_idx in (range(task_config.total_tasks)):
        with open(task_config.task_paths.format(task_idx*5), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
    return tasks

def one_hot_encode_task(task_index, num_tasks):
    encoding = np.zeros(num_tasks)
    encoding[task_index-1] = 1
    return encoding



def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b


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



def behavior(args):
    
    
    for dir in ["./models", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", args.env, args.data_mode, str(args.task))):
        os.makedirs(os.path.join("./models",args.env, args.data_mode, str(args.task)))

    if args.env == 'cheetah_vel':
        config="config/cheetah_vel/40tasks_offline.json"
        tasks = Tasks(config)
        env = HalfCheetahVelEnv(tasks)
    elif args.env == 'walker_params':
        config ="config/walker_params/50tasks_offline.json"
        tasks = Tasks(config)
        env = WalkerRandParamsWrappedEnv(tasks)
    elif args.env == 'swimmer_dir':
        config="config/swimmer_dir/50tasks_offline.json"
        tasks = Tasks(config)
        env = SwimmerDir(tasks = tasks)
    elif args.env == 'meta_world':
        tasks = TASK_SEQS['CL5']
        env = get_cl_env(tasks)
    else:
        raise RuntimeError(f'Invalid env name {args.env}')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    onehot_dim = args.num_tasks
    
    max_action = float(env.action_space.high[0])
    print("state_dim:", state_dim, "action_dim:", action_dim, "max_action:", max_action)
    env.seed(args.seed)

    yaml_path = args.config
    local_rank = args.local_rank
    use_amp = args.use_amp
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print(opt)
    opt = Config(opt)
    print("local_rank:", local_rank)
    device = "cuda:%d" % local_rank
    print("device:", device)
    args.device = device
    args.critic_mode = False

    torch.manual_seed(args.seed+local_rank)
    np.random.seed(args.seed+local_rank)
    
    
    # TODO: initialize decision diffuser model 
    save_path = os.path.join("./logs/", args.env, args.data_mode, str(args.task))
    if local_rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    dataset = Diffusion_buffer(args)
    data_len = dataset.states.shape[0]
    
    Normalizer = DatasetNormalizer(dataset, 'LimitsNormalizer')
    
    dataset.states = Normalizer.normalize(dataset.data_["states"], "states")
    dataset.actions = Normalizer.normalize(dataset.data_["actions"], "actions")
    
    if args.data_mode == "continual_diffuser": 
        dataset.init_paths()
    
    normalizers = Normalizer.normalizers
    normalizers_file = os.path.join(save_path, "normalizers.npy")
    
    with open(normalizers_file, "wb") as f:
        pickle.dump(normalizers, f)
        
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    horizon = dataset.horizon
    val_normalization = dataset.seq_val.max() # TODO: check if this is correct value were supposed to get 

    data_loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=True)

    print("data_loader_len:", len(data_loader))



    if args.env == 'meta_world':
        dim_mult = [1, 2, 2, 2]
    else: 
        dim_mult=[1, 4, 2]



    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, dim_mult=dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    
    nn_condition = ValueNNCondition(emb_dim=128, dropout=0.25)
    
    fix_mask = torch.zeros((horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    
    loss_weight = torch.ones((horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = 10.
    
    planner = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.9999,
        device=device
    )
        
    if args.task >= 2:
        
        if args.actor_load_setting is None:
            args.actor_loadpath = os.path.join("./models", str(args.env), args.data_mode, str(args.task-1), "ckpt{}.pth".format(args.
            actor_load_epoch))
                
        print("loading continual diffuser from {}...".format(args.actor_loadpath))
        planner.load(args.actor_loadpath)
        



    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(os.path.join("./logs", args.env, args.data_mode, str(args.task), current_time))
    args.writer = writer

    if args.task >= 1:
        print("training diffusion")
        
        behavior_loss = []
        tqdm_epoch = trange(opt.load_epoch + 1, args.n_behavior_epochs)
        
        planner.train()
        
        for epoch in tqdm(tqdm_epoch, disable=(local_rank != 0)):
            avg_loss = 0.
            num_items = 0
            
            pbar = data_loader
            for batch in pbar:                
                obs, act = batch["obs"]["state"].to(device), batch["act"].to(device) 
                val = batch["val"].to(device) / val_normalization
                
                x0 = torch.cat([obs, act], dim=-1)
                avg_loss += planner.update(x0=x0, condition=val)["loss"]
        
                num_items += obs.shape[0]
        
            avg_loss = avg_loss / num_items
            tqdm_epoch.set_description('Behavior Loss: {:5f}'.format(avg_loss))    
            
            if epoch % 100 == 99 and args.save_model:
                planner.save(os.path.join("./models", 
                                            str(args.env), 
                                            args.data_mode, 
                                            str(args.task), 
                                            "ckpt{}.pth".format(epoch+1)
                                            )
                                )        
            if args.writer:
                args.writer.add_scalar("actor/loss", avg_loss, global_step=epoch)
            
            behavior_loss.append(avg_loss)
        
        
        save_fig = os.path.join(save_path, "behavior_loss")
        plt.plot(np.arange(len(behavior_loss)), behavior_loss)
        plt.xlabel('behavior_loss')
        plt.ylabel('loss')
        plt.savefig(save_fig, dpi=300)
        plt.cla()
        print("behavior model finished")

    writer.close()





if __name__ == "__main__":
    
    args = get_args()
    
    print("Available CUDA devices:", torch.cuda.device_count())
    print("current local_rank:", args.local_rank)
    
    init_seeds(no=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    
    t = time.time()
    for i in range(args.starting_task, args.ending_task+1):
        args.task = i
        behavior(args)
    
    print("total time:", (time.time()-t)/3600)


