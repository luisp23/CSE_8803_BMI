import torch
import torch.nn as nn
import gym
# import d4rl
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
MAX_BZ_SIZE = 1024
soft_Q_update = True

def one_hot_encode_task(task_index, num_tasks):
    encoding = np.zeros(num_tasks)
    encoding[task_index-1] = 1
    return encoding
from typing import Union, Dict, Callable


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


class Diffusion_buffer(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        self.normalise_return = args.normalise_return
        self.data = self._load_data(args)
        
        
        self.actions = self.data["actions"].astype(np.float32)
        self.states = self.data["states"].astype(np.float32)
        self.rewards = self.data["rewards"].astype(np.float32)
        self.done = self.data["done"]
        
        
        one_hot = one_hot_encode_task(args.task, args.num_tasks)
        self.one_hot =np.repeat([one_hot], self.states.shape[0], axis = 0).astype(np.float32)

        self.data_ = {}
        self.data_["actions"] = self.actions
        self.data_["states"] = self.states

        self.returns = self.data["returns"].astype(np.float32)
        self.raw_returns = [self.returns]
        self.raw_values = []
        self.returns_mean = np.mean(self.returns)
        self.returns_std = np.maximum(np.std(self.returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            self.returns = (self.returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
            self.args.returns_mean = self.returns_mean
            self.args.returns_std = self.returns_std
        else:
            print("no normal")
        self.ys = np.concatenate([self.returns, self.actions], axis=-1)
        self.ys = self.ys.astype(np.float32)

        self.len = self.states.shape[0]
        self.data_len = self.ys.shape[0]
        self.fake_len = self.len
        
        
        
        print(self.len, "data loaded", self.data_len, "ys loaded", self.fake_len, "data faked")


        
        
        # to make compatible with clean clean diffusers 
        self.horizon = args.trajectory_horizon
        
        
        
        
    def init_paths(self, max_path_length=200): 
        
        self.o_dim, self.a_dim = self.states.shape[1], self.actions.shape[1]
        self.max_path_length = max_path_length # TODO: this is hardcoded for the swimmer_dir task, change to argument or bring in cheetah task 
        n_paths = np.sum(self.done, dtype=np.int32)
        
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths, max_path_length, 1), dtype=np.float32)
        self.seq_val = np.zeros((n_paths, max_path_length, 1), dtype=np.float32)
        
        self.indices = []
        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(self.done.shape[0]):
            if self.done[i]:
                path_lengths.append(i - ptr + 1)

                # if terminals[i] and not timeouts[i]:
                #     rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]
                #     self.tml_and_not_timeout.append([path_idx, i - ptr])

                self.seq_obs[path_idx, :i - ptr + 1] = self.states[ptr:i + 1] # TODO: need to have the observations normed (happens after init)
                self.seq_act[path_idx, :i - ptr + 1] = self.actions[ptr:i + 1]
                self.seq_rew[path_idx, :i - ptr + 1] = self.rewards[ptr:i + 1][:]

                max_start = min(path_lengths[-1] - 1, max_path_length - self.horizon)
                self.indices += [(path_idx, start, start + self.horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1


        discount = 0.99 
        
        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in range(max_path_length - 1):
            self.seq_val[:, - 2 - i] = self.seq_rew[:, -2 - i] + discount * self.seq_val[:, -1 - i]
        
        self.path_lengths = np.array(path_lengths)
        
        


    def __getitem__(self, index):

        if self.horizon == 1: # for running CuGRO
            data = self.ys[index % self.len]
            state = self.states[index % self.len]
            one_hot = self.one_hot[index % self.len]
            
            return data, state, one_hot
        
        else: # for running with decision diffusers 
             
            path_idx, start, end = self.indices[index]

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end],
                'rew': self.seq_rew[path_idx, start:end],
                'val': self.seq_val[path_idx, start],
            }

            torch_data = dict_apply(data, torch.tensor)

            return torch_data








    def __add__(self, other):
        pass






    def __len__(self):
        if self.horizon == 1: 
            return self.fake_len
        else: 
            return len(self.indices)






    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.len, size=batch_size)
        data = self.ys[indices]
        state = self.states[indices]
        one_hot = self.one_hot[indices]
        return data, state, one_hot

    def _load_data(self, args):
            dataset_path = args.dataset_path/ f'{args.env}'/f'{args.task}'/f'{args.dataset}'
            print("dataset_path:", dataset_path)
            dataset = np.load(dataset_path, allow_pickle='TRUE').item()
            data = {}
            data["states"] = dataset["observations"]
            data["actions"] = dataset["actions"]
            data["rewards"] = dataset["rewards"][:, None]
            data["done"] = dataset["dones"]
            data["returns"] = np.zeros((data["states"].shape[0], 1))
            
            
            last = 0
            for i in range(data["returns"].shape[0] - 1, -1, -1):
                last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
                data["returns"][i, 0] = last
            return data

