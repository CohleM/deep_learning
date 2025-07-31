import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
## ----------------- SETUP ----------------------------
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
backend = 'nccl' if device == 'cuda' else 'gloo'
dist.init_process_group(backend=backend,  world_size=world_size)
## ----------------- SETUP ----------------------------

inp = [torch.arange(1.,9.).reshape(2,4), torch.arange(9.,17.).reshape(2,4)]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Parameter(torch.arange(1.,5.).reshape(1,4))
    def forward(self,x):
        return x @ self.net1.T

local_data = inp[local_rank] # will get data based on its rank

model = ToyModel()
ddp_model = DDP(model)

optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-6)

for i in range(1):
    predicted = ddp_model(local_data)
    original = torch.randn_like(predicted)

    print(local_data)
    loss = (original - predicted)**2
    loss = loss.sum(dim=0)
    loss /= predicted.shape[0] 
    print('loss',loss)

    optimizer.zero_grad()

    loss.backward()

    # dist.all_reduce(model.net1.grad, op=dist.ReduceOp.SUM)
    print(f' AFTER ALL REDUCE \n RANK {dist.get_rank() }\n parameters\n { ddp_model.module.net1.grad} ')
    # do the optimizer.step
    optimizer.step()
dist.destroy_process_group()