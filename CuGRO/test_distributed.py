import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
print("heree?/")
print(f"Rank {dist.get_rank()} is running on GPU {local_rank}")