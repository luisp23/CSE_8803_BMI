from copy import deepcopy
import os
from tqdm import tqdm, trange
import functools
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, MlpScoreNet, GenerateNet, MlpGenerateNet
from utils import get_args, plot_tools, plot_successs
from dataset import Diffusion_buffer
import json
from src.envs import HalfCheetahVelEnv, WalkerRandParamsWrappedEnv, SwimmerDir
from collections import namedtuple
from continualworld.envs import get_cl_env
from continualworld.tasks import TASK_SEQS
from normalization import DatasetNormalizer

def one_hot_encode_task(task_index, num_tasks):
    encoding = np.zeros(num_tasks)
    encoding[task_index-1] = 1
    return encoding


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




def evaluate_model(args, planner, Normalizer, start_epoch=0):

    epoch = args.K*100 # NOTE: to make compatible with others; CuGRO trains until 100 epochs, which is what we are comparing to
    loss_list = []

    n_epochs = args.K*100
    tqdm_epoch = trange(start_epoch, n_epochs)

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0

        if ((epoch % 10 == 9) or epoch==0):
            for i in range(args.task):


                normalizers_file = os.path.join("./logs/", args.env, args.data_mode, str(i+1), "normalizers.npy")
                with open(normalizers_file, "rb") as f:
                    gene_normalizers = pickle.load(f)
                Normalizer.normalizers = gene_normalizers
                
                # TODO: change to take in trained diffusion model     
                envs = args.eval_func(planner, task=i+1, Normalizer=Normalizer)
                
                    
                env_returns = [envs[i].dbag_return for i in range(args.seed_per_evaluation)]
                mean = np.mean(env_returns)
                std = np.std(env_returns)
                
                # if args.env == "meta_world": # TODO: probably will not evaluate on this
                #     env_success = [envs[i].pop_successes() for i in range(args.seed_per_evaluation)]
                #     success_mean = np.mean(env_success)
                #     success_std = np.std(env_success)
                #     success_list[i].append(np.array([success_mean, success_std, epoch + (args.task-1)*n_epochs]))
                #     all_success_list[i].append(np.array(env_success))
                #     print("success/rew:", success_mean, "success/std", success_std)
                
                
                if args.writer:
                    args.writer.add_scalar("eval/rew", mean, global_step=epoch)
                    args.writer.add_scalar("eval/std", std, global_step=epoch)
                
                print("eval/rew:", mean, "eval/std", std)
                
                return_list[i].append(np.array([mean, std, epoch + (args.task-1)*n_epochs])) 
                all_return_list[i].append(np.array(env_returns))
        
    return loss_list


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




    # NOTE: the policy_fn passed when training CuGRO
    # def select_actions(self, states, sample_per_state=32, select_per_state=1, alpha=100, replace=False, weighted_mean=False, diffusion_steps=25, Normalizer=None):
    #     returns, actions = self.sample(states, sample_per_state, diffusion_steps, Normalizer=Normalizer)
    #     if isinstance(select_per_state, int):
    #         select_per_state = [select_per_state] * actions.shape[0]
    #     if (isinstance(alpha, int) or isinstance(alpha, float)):
    #         alpha = [alpha] * actions.shape[0]
    #     if (isinstance(replace, int) or isinstance(replace, float) or isinstance(replace, bool)):
    #         replace = [replace] * actions.shape[0]
    #     if (isinstance(weighted_mean, int) or isinstance(weighted_mean, float) or isinstance(weighted_mean, bool)):
    #         weighted_mean = [weighted_mean] * actions.shape[0]
    #     # select `select_per_sample` data from 32 data, ideally should be 1.
    #     # Selection should happen according to `alpha`
    #     # replace defines whether to put back data
    #     out_actions = []
    #     for i in range(actions.shape[0]):
    #         raw_actions = self._select(returns[i], actions[i], alpha=alpha[i], num=select_per_state[i], replace=replace[i])
    #         out_actions.append(np.average(raw_actions, weights=self.weighted if weighted_mean[i] else None, axis=0))
    #     return out_actions




def pallarel_eval_policy(planner, env_name, task, seed, eval_episodes=20, diffusion_steps=15, Normalizer=None, args=None, state_dim=20, action_dim=6):
        
    eval_envs = []
    
    for i in range(eval_episodes):
        if env_name == 'cheetah_vel':
            config="config/cheetah_vel/40tasks_offline.json"
            tasks = Tasks(config)
            env = HalfCheetahVelEnv(tasks)
            env.set_task_idx(task-1)
        elif env_name == 'walker_params':
            config ="config/walker_params/50tasks_offline.json"
            tasks = Tasks(config)
            env = WalkerRandParamsWrappedEnv(tasks)
            env.set_task_idx(task-1)
        elif args.env == 'swimmer_dir':
            config="config/swimmer_dir/50tasks_offline.json"
            tasks = Tasks(config)
            env = SwimmerDir(tasks = tasks)
            env.set_task_idx(task-1)
        elif args.env == 'meta_world':
            tasks = TASK_SEQS['CL5']
            env = get_cl_env(tasks)
            env.set_task_idx(task-1)
        else:
            raise RuntimeError(f'Invalid env name {env_name}')

        eval_envs.append(env)
        env.seed(seed + 1001 + i)
        env.dbag_state = env.reset()
        env.dbag_return = 0.0
        env.alpha = 100 # 100 could be considered as deterministic sampling since it's now extremely sensitive to normalized Q(s, a)
        # env.select_per_state = select_per_state # NOTE: necessary? 
            
    ori_eval_envs = [env for env in eval_envs]
    t = time.time()


    solver = "ddpm"
    sampling_step = 5
    target_return = 0.001
    w_cfg = 1.2
    
    
    num_envs = len(eval_envs)
    prior = torch.zeros((num_envs, args.trajectory_horizon, state_dim + action_dim), device=args.device)
    condition = torch.ones((num_envs, 1), device=args.device) * target_return
    
    
    while len(eval_envs) > 0:
        new_eval_envs = []
        
        states = np.stack([env.dbag_state for env in eval_envs])
        states = Normalizer.normalize(states, 'states') # (num_envs, state_dim)
        
        obs = torch.tensor(states, device=args.device, dtype=torch.float32)
        
        # actions = policy_fn(
        #     states, 
        #     sample_per_state=32, 
        #     select_per_state=[env.select_per_state for env in eval_envs], 
        #     alpha=[env.alpha for env in eval_envs], 
        #     replace=False, 
        #     weighted_mean=False, 
        #     diffusion_steps=diffusion_steps, 
        #     Normalizer=Normalizer
        # )
        # actions = Normalizer.unnormalize(np.array(actions), 'actions')
        
        # actions: (len(10)) 
        # actions[0]: (action_dim,)
        # TODO: for every envrionment, sample a horizon; take out the action from the horizon
        actions = []
        
        
        
        # sample trajectories
        prior[:, 0, :state_dim] = obs
        
        # TODO: are these the right parameters for the task/domain? check CleanDiffuser pipelines/
        traj, log = planner.sample(
            prior, 
            solver=solver,
            n_samples=num_envs, 
            sample_step_schedule="quad_continuous",
            sample_steps=sampling_step, 
            use_ema=True,
            condition_cfg=condition, 
            w_cfg=w_cfg, 
            temperature=1.0
        )
        
        # TODO: why do these need to be clipped?  this was with halfcheetah mujoco task
        # actions = traj[:, 0, state_dim:].clip(-1., 1.).cpu().numpy()
        actions = traj[:, 0, state_dim:].cpu().numpy()


        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])  
            env.dbag_return += reward
            env.dbag_state = state
            if not done:
                new_eval_envs.append(env)
        
        eval_envs = new_eval_envs
    

    print("time:", time.time() - t)
    
    return ori_eval_envs






def critic(args):
    
    for dir in ["./models", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    if not os.path.exists(os.path.join("./models", args.env, args.data_mode, str(args.task))):
        os.makedirs(os.path.join("./models", args.env,  args.data_mode, str(args.task)))
    
    if not os.path.exists(os.path.join("./logs/" + str(args.l), args.env, args.data_mode, str(args.task))):
        os.makedirs(os.path.join("./logs/" + str(args.l), args.env, args.data_mode, str(args.task)))

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



    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    horizon = args.trajectory_horizon
    onehot_dim = args.num_tasks
    
    max_action = float(env.action_space.high[0])
    print("state_dim:", state_dim, "action_dim:", action_dim, "max_action:", max_action)

    
    # TODO: need to make this compatible with our policy, or just bring in into the evalute function above ?
    args.eval_func = functools.partial(
        pallarel_eval_policy, 
        env_name=args.env, 
        task=args.task, 
        seed=args.seed, 
        eval_episodes=args.seed_per_evaluation, 
        diffusion_steps=args.diffusion_steps,
        args=args,
        state_dim=state_dim,
        action_dim=action_dim, 
        
    )
    
    
    
    
    
    
    

    

    
    
    # TODO: initialize models 
        
    if args.actor_load_setting is None:
        args.actor_loadpath = os.path.join("./models", str(args.env),  args.data_mode, str(args.task), "ckpt{}.pth".format(args.actor_load_epoch))



    nn_diffusion = JannerUNet1d(
        state_dim + action_dim, dim_mult=[1, 4, 2],
        timestep_emb_type="positional", attention=False, kernel_size=5)
    
    nn_condition = ValueNNCondition(emb_dim=128, dropout=0.25)

    fix_mask = torch.zeros((horizon, state_dim + action_dim))
    fix_mask[0, :state_dim] = 1.
    
    loss_weight = torch.ones((horizon, state_dim + action_dim))
    loss_weight[0, state_dim:] = 10.


    planner = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.9999,
        device=args.device
    )



    planner.load(args.actor_loadpath)
    planner.eval()

    fix_mask = torch.zeros((horizon, state_dim + action_dim))
    fix_mask[0, :state_dim] = 1.
    
    loss_weight = torch.ones((horizon, state_dim + action_dim))
    loss_weight[0, state_dim:] = 10.


    planner = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.9999,
        device=args.device)






    # print("loading actor{}...".format(args.actor_loadpath))

    
    dataset = Diffusion_buffer(args)
    Normalizer = DatasetNormalizer(dataset, 'LimitsNormalizer')
    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # data_len = dataset.states.shape[0]
   


    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(os.path.join("./logs", args.env, args.data_mode+str(args.diffusion_steps), str(args.task), current_time))
    args.writer = writer
    
    evaluate_model(args, planner, Normalizer)
    



if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    global return_list
    global all_return_list
    global success_rate_list

    return_list = [[] for _ in range(args.num_tasks)]
    all_return_list = [[] for _ in range(args.num_tasks)]
    if args.env == "meta_world":
        success_list = [[] for _ in range(args.num_tasks)]
        all_success_list = [[] for _ in range(args.num_tasks)]

    # Serialize and save list to file
    def save_list_to_file(file_path, my_list):
        with open(file_path, 'wb') as f:
            pickle.dump(my_list, f)

    # Load and deserialize list from file
    def load_list_from_file(file_path):
        with open(file_path, 'rb') as f:
            my_list = pickle.load(f)
        return my_list

    for dir in ["./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    file_path = os.path.join("./logs/" + str(args.l), args.env, args.data_mode)
    
    for i in range(1, args.ending_task+1):
        
        args.task = i
        
        if args.task >=2:
            return_file = os.path.join(file_path, str(args.task-1), "return_list.npy")
            return_list = load_list_from_file(return_file)
            all_return_file = os.path.join(file_path, str(args.task-1), "all_return_list.npy")
            all_return_list = load_list_from_file(all_return_file)
            
            
            # if args.env == "meta_world": # NOTE: leaving for if time to evaluate on this domain 
            #     success_file = os.path.join(file_path, str(args.task-1), "success_list.npy")
            #     success_list = load_list_from_file(success_file)
            #     all_success_file = os.path.join(file_path, str(args.task-1), "all_success_list.npy")
            #     all_success_list = load_list_from_file(all_success_file)

        critic(args)

        return_file = os.path.join(file_path, str(args.task), "return_list.npy")
        save_list_to_file(return_file, return_list)
        all_return_file = os.path.join(file_path, str(args.task), "all_return_list.npy")
        save_list_to_file(all_return_file, all_return_list)
        
        
        # if args.env == "meta_world": # NOTE: leaving for if time to evaluate on this domain
        #     success_file = os.path.join(file_path, str(args.task), "success_list.npy")
        #     save_list_to_file(success_file, success_list)
        #     all_success_file = os.path.join(file_path, str(args.task), "all_success_list.npy")
        #     save_list_to_file(all_success_file, all_success_list)

        
        
        plot_tools(
            folder_name= "./logs/" + str(args.l), 
            env_name=args.env, 
            task=args.task, 
            return_list=return_list, 
            all_return_list=all_return_list, 
            data_mode=args.data_mode
        )
        
        
        # if args.env == "meta_world":
        #     plot_successs(folder_name= "./logs/" + str(args.l), env_name = args.env, task = args.task, return_list = success_list, all_return_list = all_success_list, data_mode=args.data_mode)

        
        
        