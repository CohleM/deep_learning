from torch.testing._internal.common_distributed import spawn_threads_and_init_comms

from transformers import AutoModelForCausalLM, AutoTokenizer
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

# from llama2_model import Transformer, ModelArgs

import torch.distributed as dist
import torch
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor

# from llama2_model import Transformer, ModelArgs
from torch.distributed._composable.fsdp import fully_shard
from qwen_monkey_patch import apply_qwen_patches
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp._runtime_utils import _lazy_init

## NOTE : We need to also make cos and sin separate from position_embeddings in the modeling_qwen.py file. 
# something like this 

# class Qwen2Attention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: Qwen2Config, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = self.head_dim**-0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = True
#         self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
#         self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
#         self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
#         self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cos: torch.Tensor,sin: torch.Tensor, <---------------- see here
#         attention_mask: Optional[torch.Tensor],
#         past_key_value: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:




# def prepare_llama_tp_layer(layer, device_mesh):

#     parallelize_plan = {
#         "input_layernorm": SequenceParallel(),
#         "self_attn" : PrepareModuleInput(
#         input_kwarg_layouts = {"hidden_states" : Shard(1), "cos": Replicate(), "sin": Replicate(),"attention_mask" : Shard(1) },
#         desired_input_kwarg_layouts = {"hidden_states" : Replicate(), "cos": Replicate(), "sin": Replicate(), "attention_mask": Replicate()}

#     ),
#         "self_attn.q_proj": ColwiseParallel(use_local_output=False),
#         "self_attn.k_proj": ColwiseParallel(use_local_output=False),
#         "self_attn.v_proj": ColwiseParallel(use_local_output=False),
#         "self_attn.o_proj": RowwiseParallel(
#             output_layouts=Shard(1)
#         ),
#         "post_attention_layernorm": SequenceParallel(),
#         "mlp.gate_proj": ColwiseParallel(),
#         "mlp.up_proj": ColwiseParallel(),
#         "mlp.down_proj": RowwiseParallel(
#             output_layouts=Shard(1)
#         )
#     }
#     parallelize_module(
#         module=layer,
#         device_mesh=device_mesh,
#         parallelize_plan=parallelize_plan
#     )


# def prepare_tp_model(model, mesh):
#     for layer in model.model.layers:
#         prepare_llama_tp_layer(layer, mesh['TP'])

    
# # ----- outer block --------
#     parallelize_plan = {
#         "model.embed_tokens": RowwiseParallel(
#             input_layouts=Replicate(),
#             output_layouts=Shard(1)
#         ),
#         "model.norm": SequenceParallel(),
#         "lm_head": ColwiseParallel(input_layouts=(Shard(1))) # we are just specifying what it's current input layout is but internally it'll convert that Shard(1) to Replicate(), and the output will be Shard(-1)
#     }
#     parallelize_module(
#         module=model,
#         device_mesh=mesh['TP'],
#         parallelize_plan=parallelize_plan
#     )
#     return model

def prepare_llama_tp_layer(layer, device_mesh):

    parallelize_plan = {
        
        "self_attn" : PrepareModuleInput(
        input_kwarg_layouts = {"hidden_states" : Replicate(), "cos": Replicate(), "sin": Replicate(),"attention_mask" : Replicate() },
        desired_input_kwarg_layouts = {"hidden_states" : Replicate(), "cos": Replicate(), "sin": Replicate(), "attention_mask": Replicate()}

    ),
        "self_attn.q_proj": ColwiseParallel(use_local_output=False),
        "self_attn.k_proj": ColwiseParallel(use_local_output=False),
        "self_attn.v_proj": ColwiseParallel(use_local_output=False),
        "self_attn.o_proj": RowwiseParallel(
            # output_layouts=Shard(1)
        ),
        # "post_attention_layernorm": SequenceParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(
            # output_layouts=Shard(1)
        )
    }
    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )


def prepare_tp_model(model, mesh):
    for layer in model.model.layers:
        prepare_llama_tp_layer(layer, mesh['TP'])

    
# ----- outer block --------
    parallelize_plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            # output_layouts=Shard(1)
        ),
        # "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel() # we are just specifying what it's current input layout is but internally it'll convert that Shard(1) to Replicate(), and the output will be Shard(-1)
    }
    parallelize_module(
        module=model,
        device_mesh=mesh['TP'],
        parallelize_plan=parallelize_plan
    )
    return model


def prepare_dp_model(model, mesh):

    def get_module_cls_from_name(name):
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__

    transformer_layer_cls = {
        get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=mesh['DDP', 'FSDP'],
        device_id=torch.cuda.current_device()
    )



# for transformers llama model
# def prepare_tp_model(model, mesh):

#     layer_tp_plan = {
#         # Now the input and output of SequenceParallel has Shard(1) layouts,
#         # to represent the input/output tensors sharded on the sequence dimension
#         ### ------ The module names will be different for new models -----, take a look at model.named_parameters()
#         "attention_norm": SequenceParallel(),
#         "attention": PrepareModuleInput(
#             input_layouts=(Shard(1), Replicate()),
#             desired_input_layouts=(Replicate(), Replicate()),
#         ),
#         "attention.wq": ColwiseParallel(use_local_output=False),
#         "attention.wk": ColwiseParallel(use_local_output=False),
#         "attention.wv": ColwiseParallel(use_local_output=False),
#         "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
#         "ffn_norm": SequenceParallel(),
#         "feed_forward": PrepareModuleInput(
#             input_layouts=(Shard(1),),
#             desired_input_layouts=(Replicate(),),
#         ),
#         "feed_forward.w1": ColwiseParallel(),
#         "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
#         "feed_forward.w3": ColwiseParallel(),
#     }


#     # Apply TP
#     for layer_id, transformer_block in enumerate(model.layers):

#         parallelize_module(
#             module=transformer_block,
#             device_mesh=mesh['TP'],
#             parallelize_plan=layer_tp_plan,
#         )

#     model = parallelize_module(
#         model,
#         mesh['TP'],
#         {
#             "tok_embeddings": RowwiseParallel(
#                 input_layouts=Replicate(),
#                 output_layouts=Shard(1),
#             ),
#             "norm": SequenceParallel(),
#             "output": ColwiseParallel(
#                 input_layouts=Shard(1),
#                 output_layouts=Replicate()
#             ),
#         }
#     )
    
#     return model



from dataclasses import dataclass

@dataclass
class Config:
    train_batch_size: int = 64
    rollout_n: int = 8
    mini_batch_size: int = 8
    model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    ddp_size: int = 2
    tp_size: int = 2
    lr: float = 1e-6
#     train_batch_size: int = 64
        
config = Config()



def main():
    apply_qwen_patches()
    # can't use os.environ['LOCAL_RANK'] in spawn_threads_and_init_comms so using dist.get_rank() which gives the
    # local rank, but instead we must to os.environ['LOCAL_RANK'] otherwise.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_device_mesh = init_device_mesh(device, (1,1,2), mesh_dim_names = ['DDP','FSDP','TP'])
    

    # In your main() function, add this before preparing the model:
    model = AutoModelForCausalLM.from_pretrained(config.model_name, attn_implementation="eager").to('cuda')

    model = prepare_tp_model(model, model_device_mesh)

    model = prepare_dp_model(model, model_device_mesh)

    torch.manual_seed(42)
    import pickle

    with open('data_list.pkl', 'rb') as f:
        data_list = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    ## --- new code ---- 
    input_ids = [item['states'] for item in data_list]
    action_input_ids = [item['actions'] for item in data_list]

    batch = {"input_ids": input_ids}
    action_batch = {"input_ids": action_input_ids}

    padded_input_ids = tokenizer.pad(batch, padding=True, padding_side='left') # make every row in the batch to have same length
    action_input_ids = tokenizer.pad(action_batch, padding=True, padding_side='left')['input_ids'].to('cuda')

    # print(f'rank {dist.get_rank()} and  padded input ids shape {padded_input_ids['input_ids'].shape} attention mask shape {padded_input_ids['attention_mask'].shape} ')
    attention_mask = padded_input_ids['attention_mask'].to('cuda')
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill_(attention_mask == 0, 0).to('cuda')


    with torch.no_grad():
        out = model(input_ids=padded_input_ids['input_ids'].to('cuda'), attention_mask=attention_mask, position_ids=position_ids).logits
    dist.barrier() 
    
    # Change this line - use log_softmax instead of softmax

    # print(out[0][0][:5])
    print(f' rank {dist.get_rank()} out shape {out.shape} just out ',out[-1][0][:5], '\n\n')

    ## --- new code ---- 


    # input_ids = [item['states'] for item in data_list]
    # cross_entorpy = torch.nn.CrossEntropyLoss(reduction='none')
    # loss = (cross_entorpy(out.view(B*T, -1), y.view(B*T)).view(B,T) * attention_mask).sum(dim=-1)/(attention_mask.sum(dim=-1))
    # print(f'rank {dist.get_rank()} , out shape, {out.shape}')

    # # ------- new model ---------
    # new_model = AutoModelForCausalLM.from_pretrained(config.model_name, attn_implementation="eager", torch_dtype=torch.bfloat16).to('cuda') 
    # input_ids = [item['states'] for item in data_list]
    # action_input_ids = [item['actions'] for item in data_list]

    # batch = {"input_ids": input_ids}
    # action_batch = {"input_ids": action_input_ids}

    # padded_input_ids = tokenizer.pad(batch, padding=True, padding_side='left') # make every row in the batch to have same length
    # action_input_ids = tokenizer.pad(action_batch, padding=True, padding_side='left')['input_ids'].to('cuda')

    # # print(f'rank {dist.get_rank()} and  padded input ids shape {padded_input_ids['input_ids'].shape} attention mask shape {padded_input_ids['attention_mask'].shape} ')
    # attention_mask = padded_input_ids['attention_mask'].to('cuda')
    # position_ids = attention_mask.long().cumsum(-1) - 1
    # position_ids = position_ids.masked_fill_(attention_mask == 0, 0).to('cuda')


    # with torch.no_grad():
    #     new_out = new_model(input_ids=padded_input_ids['input_ids'].to('cuda'), attention_mask=attention_mask, position_ids=position_ids).logits
    # # Change this line - use log_softmax instead of softmax

    # # print(out[0][0][:5])
    # print(f' rank {dist.get_rank()} new out', new_out[-1][0][:5], '\n\n')
    

    # new_loss = cross_entorpy(new_out.view(B*T, -1), y_new.view(B*T)).view(B,T).mean(dim=-1)

    # print(loss, new_loss)

    return
main()