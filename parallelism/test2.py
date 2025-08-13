import os
import torch
from dataclasses import dataclass
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from datasets import load_dataset
import math
import copy
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader


class RLDataset(Dataset):

    def __init__(self, data_path, responses_per_prompt):

        self.dataset = load_dataset(data_path, split='train')
        self.responses_per_prompt = responses_per_prompt

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        answer = ex["answer"]

        return {
            "messages": messages,
            "answer": answer
        }
    def __len__(self):
        return len(self.dataset)



    def collate_fn(self, batch):

        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]



def setup():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def start():
    
    device = "cpu" 
    mesh = init_device_mesh(device, (2,2,2), mesh_dim_names = ['DDP','FSDP', 'TP'])

    train_data = RLDataset("CohleM/olympiad_small", 4)
    train_dataloader = StatefulDataLoader(train_data, batch_size=3, drop_last=True, collate_fn=train_data.collate_fn)
    # construct train dataloader
    
    for train_batch in train_dataloader:
        if mesh['TP'].get_local_rank() == 0:
            # then split across ddp
            data_list = split_data_list(train_batch, mesh=mesh['DDP'])
        
            print(f'rank {dist.get_rank()} data len {len(data_list)} first element {data_list[0]} \n')
            break

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