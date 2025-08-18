import os
import torch
from dataclasses import dataclass
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
# from datasets import load_dataset
import math
import copy
# from torch.utils.data import Dataset
# from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed._composable.fsdp import fully_shard
# from transformers import AutoTokenizer, AutoModel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput
# from llama2_model import Transformer, ModelArgs


def setup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' 
    backend = 'nccl' if device == 'cuda' else 'gloo'
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if device == 'cuda':
        torch.cuda.set_device(local_rank)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Initialize with explicit parameters
    dist.init_process_group(
        backend=backend, 
        world_size=world_size,
        rank=rank
    )


def split_data_list(data_list, mesh):
  # we need to scatter this data_list across this mesh group, from local group 0.

    rank = mesh.get_local_rank()
    size = mesh.size()

    if rank == 0:
        data_per_ddp = math.ceil(len(data_list)/size)
    
    lists = [data_list[i * data_per_ddp: (i+1)* data_per_ddp] if rank ==0 else None for i in range(size)]
    
    lst = [None] # this is the output list
    dist.scatter_object_list(lst, lists, src=None, group_src=0, group=mesh.get_group())

    # print(f'rank {dist.get_rank()} got this list{lst}' )
    return lst[0]


def gather_data_list(data_list, mesh):
    # we need to scatter this data_list across this mesh group, from local group 0.

    rank = mesh.get_local_rank()
    size = mesh.size()

    lists = [None for i in range(size)] if rank==0 else None # Must be None on non-dst ranks otherwise it will call dist.gather_object in other ranks as well is None, it will be called only in the group_dst rank


    dist.gather_object(data_list,lists, group_dst=0, group=mesh.get_group()) 

    #   print(f'rank {dist.get_rank()} got this list{lists}' )
    return lists



def check_mem_allocated(rank, msg):
    ans = torch.cuda.memory_allocated() / (1024**3)
    print(f'RANK {rank} MEMORY_ALLOCATED {msg} {ans}')

import torch.nn as nn

def start():

    data_list = [torch.ones((2,2)), torch.ones((2,2))*2]
    device = "cpu" 
    model_device_mesh = init_device_mesh(device, (1,2,2), mesh_dim_names = ['DDP','FSDP','TP'])
    
    device_mesh = init_device_mesh(device, (2,2), mesh_dim_names = ['dp', 'tp'])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(4,4,bias=False)
        def forward(self,x):
            return self.layer0(x)
        
    model = Model()

    model.layer0.weight = nn.Parameter(torch.arange(1.,17.).reshape(4,4))
    
    parallelize_module(
        module=model,
        device_mesh=model_device_mesh['TP'],
        parallelize_plan={
            "layer0": RowwiseParallel() # since rowwise parallel will replicate the output layouts, each tp rank will get the same outptu.
        }
    )

    data_list = [torch.ones((2,2)), torch.ones((2,2))*2]
    x = split_data_list(data_list, device_mesh['dp'])[0]


    fully_shard(model.layer0, mesh=model_device_mesh['DDP','FSDP'])
    fully_shard(model, mesh=model_device_mesh['DDP','FSDP'])
    
    optim = torch.optim.AdamW(model.parameters(), lr=1e-6)


    optim.zero_grad(set_to_none=True)
    out = model(x)
    print(out)
    dist.barrier()
    torch.manual_seed(42)
    original = torch.randn_like(out)
    loss = (original - out)**2
    loss = loss.sum()
    
#     print(f'rank {dist.get_rank()} loss {loss} ')
    
    print(f' RANK {dist.get_rank()} x = {x} weights = {model.layer0.weight} \n\n')
    loss.backward()

    
#     print(f' RANK {dist.get_rank()} x = {x} \n\n')
#     print(f' RANK {dist.get_rank()} x = {x} weights = {model.layer0.weight.grad} \n\n')
    
     

    return

@dataclass
class Config:
    train_batch_size: int = 64
    mini_batch_size: int = 8
    model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    ddp_size: int = 2 
    tp_size: int = 1 
    lr: float = 1e-6
    data_path: str = 'CohleM/olympiad_small'
    responses_per_prompt: int = 4
    per_rollout_size: int = 3
#     train_batch_size: int = 64
#  
def main():
    # setup process groups.
    setup()

    # Initialize ppo trainer with some config
    start()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()