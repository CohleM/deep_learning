import os
import torch
import os
from torch import distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput
from torch.distributed._tensor import Shard, Replicate, distribute_tensor, DTensor

from llama2_model import Transformer, ModelArgs


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

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()

        self.w1 = nn.Linear(2,4, bias=False)
        self.w3 = nn.Linear(2,4, bias=False)
        self.w2 = nn.Linear(4,2, bias=False)

        with torch.no_grad():
            self.w1.weight = nn.Parameter(torch.arange(1., 9.).reshape(4,2)) # cause weight will have shape opposite of whats specified in nn.Linear()
            self.w3.weight = nn.Parameter(torch.arange(9., 17.).reshape(4,2))
            self.w2.weight = nn.Parameter(torch.tril(torch.ones(2,4), diagonal=-1))

    def forward(self, x):
        return self.w2(self.w1(x) * self.w3(x))

model = ToyModel()

tp_plan = {
    "w1" : ColwiseParallel(),
    "w2" : RowwiseParallel(),
    "w3" : ColwiseParallel()
}

model = parallelize_module(model, mesh['TP'], tp_plan)
x = distribute_tensor(torch.ones(3, 2), mesh['TP'], placements=[Replicate()]) # this is simply a tensor because ColwiseParallel will automatically convert it's input from torch.Tensor to torch.DTensor
out = model(x)


print(f'RANK {dist.get_rank()} w1.weight \n {model.w1.weight}, \n MM values \n{model.w1(x) * model.w3(x)} output values \n{out} \n')


# print(f'RANK {dist.get_rank()} Y local {y_local} \n MM_local {x_local @ y_local.T}')
# print(f'Global rank : {dist.get_rank()} \n\n WEIGGHT\n {model.net1.weight} \n OUTPUT\n {out}')
