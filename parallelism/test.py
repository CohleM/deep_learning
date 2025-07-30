import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from torch.distributed._tensor import Shard, Replicate, distribute_tensor, DTensor
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import torch

import torch
import os
from torch import distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
from llama2_model import Transformer, ModelArgs
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
)


# import warnings # ignore all warning messages
# warnings.filterwarnings("ignore")

# Code for helping init process group
device = 'cuda' if torch.cuda.is_available() else 'cpu'
backend = 'nccl' if device == 'cuda' else 'gloo'

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

# print('local rank', local_rank)

dist.init_process_group(backend=backend, world_size=world_size)

if device == 'cuda':
    torch.cuda.set_device(local_rank)


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(8, 8, bias=False)
        self.relu = nn.ReLU()

        with torch.no_grad():
            self.net1.weight = nn.Parameter(torch.arange(1., 65.).reshape(8,8))
            print(self.net1.weight)
    def forward(self, x):
        return self.relu(x @ self.net1.weight.T)

model = ToyModel()



simple_llama2_config = ModelArgs(dim=16, n_layers=1, n_heads=4, vocab_size=64)

# model = Transformer.from_model_args(simple_llama2_config).to(device)

# """
# mesh = init_device_mesh('cpu', (2, 4), mesh_dim_names=["FSDP", "TP"])

mesh = init_device_mesh('cpu', (2,), mesh_dim_names=["TP"])

# layer_tp_plan = {
#     # by default ColwiseParallel input layouts is replicated
#     # and RowwiseParallel output layouts is replicated
#     # "attention_norm" : SequenceParallel(),
#     "attention.wq": ColwiseParallel(use_local_output=False),
#     "attention.wk": ColwiseParallel(use_local_output=False),
#     "attention.wv": ColwiseParallel(use_local_output=False),
#     "attention.wo": RowwiseParallel(),
#     "feed_forward.w1": ColwiseParallel(),
#     "feed_forward.w2": RowwiseParallel(),
#     "feed_forward.w3": ColwiseParallel(),
# }
# Apply TP

# model = parallelize_module(
#     model,
#     mesh['TP'],
#     {
#         "net1": ColwiseParallel(),
#     }
# )

# for layer_id, transformer_block in enumerate(model.layers):
#     # layer_tp_plan = {...}  # i.e. the plan we just generated

#     parallelize_module(
#         module=transformer_block,
#         device_mesh=mesh['TP'],
#         parallelize_plan=layer_tp_plan,
#     )

# model = parallelize_module(
#     model,
#     mesh['TP'],
#     {
#         "tok_embeddings": RowwiseParallel(
#             input_layouts=Replicate(),
#         ),
#         "output": ColwiseParallel(
#             # output_layouts=Replicate(),
#         ),
#     }
# )


# model = parallelize_module(
#     model.to('cpu'),
#     mesh['TP'],
#     layer_tp_plan
# )
# model = fully_shard(model, mesh=mesh["FSDP"])

# # Do the operation

# x_global = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1).expand(1, 8, 16)  # [B, S, H]

# # Create the sharded tensor explicitly
# x_local = DTensor.from_local(
#     x_global,
#     device_mesh=mesh["TP"],
#     placements=[Replicate()]          # shard sequence dim
# )

# out = model.layers[0].attention_norm(x_local)
# out = model.
# print(model)
# print(f'Global rank: {dist.get_rank()} \n input\n {x} \n\n output\n {out}')


# x = torch.arange(1.,33.).reshape(2,16)

# x = torch.arange(1.,17.).reshape(1,16)
# x = torch.arange(1,17).reshape(1,16)
# x_local = DTensor.from_local(
#     x,
#     device_mesh=mesh["TP"],
#     placements=[Replicate()]          # shard sequence dim
# )
# out = model(x_local)
# out = model.layers[0].attention_norm(x)
# wq = model.layers[0].attention.wq.weight

# print(model)

# out = model.tok_embeddings(x)
# print(f'Global rank: {dist.get_rank()},  Weights array: {wq} \n\n')

# print(f'Global rank: {dist.get_rank()}, shape {model.layers[0].attention.wq.weight.to_local().shape} Weights array: {list(model.named_parameters())} \n\n')
# print(f'Global rank: {dist.get_rank()}, shape {model.output.weight.to_local().shape} Weights array: {model.output.weight} \n\n OUTPUT:\n {out}')
# print(f'Global rank: {dist.get_rank()}, shape {model.tok_embeddings.weight.to_local().shape} Weights array: {model.tok_embeddings.weight} \n\n OUTPUT:\n {out}')

# x = torch.randn(1, 8, 16)       # plain tensor
# out = model.layers[0].attention_norm(x)


x = torch.randn((4,8))
y = torch.randn((4,8))
x_local = distribute_tensor(x, device_mesh = mesh['TP'], placements=[Replicate()])

y_local = distribute_tensor(y, mesh['TP'], placements=[Shard(0)])

print(f'Y local {y_local} \n x_local {x_local @ y_local.T}')
# print(f'Global rank : {dist.get_rank()} \n\n WEIGGHT\n {model.net1.weight} \n OUTPUT\n {out}')
