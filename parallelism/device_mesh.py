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

model = Transformer.from_model_args(simple_llama2_config).to(device)
print(model)
# model.init_weights()

# """
mesh = init_device_mesh('cpu', (1, 2, 2), mesh_dim_names=["DDP", "FSDP", "TP"])
# mesh = init_device_mesh('cpu', (4,), mesh_dim_names=["TP"])
print(f' RANK {dist.get_rank() } MESH', mesh['TP'])

layer_tp_plan = {
    # Now the input and output of SequenceParallel has Shard(1) layouts,
    # to represent the input/output tensors sharded on the sequence dimension
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
# Apply TP
for layer_id, transformer_block in enumerate(model.layers):
    # layer_tp_plan = {...}  # i.e. the plan we just generated

    parallelize_module(
        module=transformer_block,
        device_mesh=mesh['TP'],
        parallelize_plan=layer_tp_plan,
    )

model = parallelize_module(
    model,
    mesh['TP'],
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            # output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)


x = torch.arange(1,17).reshape(1,16)
out = model.tok_embeddings(x)
# # out = model.layers[0].feed_forward.w1(x)
# # out = model.layers[0].attention.wq(x)

# # import time
# # time.sleep(5)
print(f'Global rank: {dist.get_rank()}, \n\n OUTPUT:\n {model.tok_embeddings.weight.shape}')
