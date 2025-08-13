
import os
import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh


import nest_asyncio

from transformers import AutoModel

from transformers import AutoModelForCausalLM

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import (DTensor, Replicate, Shard,
                                       distribute_tensor)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)


from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_state_dict
)




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

def prepare_llama_tp_layer(layer, device_mesh):

    parallelize_plan = {
        "input_layernorm": SequenceParallel(),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1)
        ),
        "post_attention_layernorm": SequenceParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(
            output_layouts=Shard(1)
        )
    }
    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_environment_variables(mesh):
    if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
        del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
    monkey_patch_torch_reductions()
    cuda_visible_devices = mesh["TP"].size() * [None]
    dist.all_gather_object(
        cuda_visible_devices,
        os.environ["LOCAL_RANK"],
        mesh["TP"].get_group()
    )
    # print(f' GLOBAL RNAK {dist.get_rank()} devices {cuda_visible_devices} ')
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

def start():
    # nest_asyncio.apply()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # local_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)
    # torch.cuda.synchronize()
    mesh = init_device_mesh(device, (2,2,2), mesh_dim_names = ['DDP','FSDP', 'TP'])
    print(f"GLOBAL RANK {dist.get_rank()} mesh {mesh['DDP']}")
    return

        
def main():
    setup()
    start()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()