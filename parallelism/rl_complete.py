
import os
import random
import torch
import asyncio
import pickle
from dataclasses import dataclass
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.entrypoints.engine import Engine
import sglang as sgl
import nest_asyncio
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from torchdata.stateful_dataloader import StatefulDataLoader
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
# from transformers import AutoTokenizer, AutoModel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput
# from llama2_model import Transformer, ModelArgs
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp._runtime_utils import _lazy_init

# from data import RLDataset

from datasets import load_dataset
import copy
# from RL2.datasets.base import BaseDataset, load_dataset
from torch.utils.data import Dataset

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
def load_model_to_device(worker, device):
    
    if not getattr(worker.config, "offload_model", False):
        return

    _lazy_init(worker.model, worker.model)
    for handle in worker.model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(device, non_blocking=True)
        flat_param._local_shard = flat_param.data



def setup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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




# works only for llama models for now.
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


def prepare_tp_model(model, mesh):
    for layer in model.model.layers:
        prepare_llama_tp_layer(layer, mesh['TP'])

    
# ----- outer block --------
    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel()
    }
    parallelize_module(
        module=model,
        device_mesh=mesh['TP'],
        parallelize_plan=parallelize_plan
    )
    return model

# def prepare_dp_model(model, mesh):
#     for layer in model.model.layers:
#         fully_shard(layer, mesh=mesh['DDP', 'FSDP'])
    
#     sharded_model = fully_shard(model, mesh=mesh['DDP', 'FSDP'])
#     return sharded_model

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

class Worker:
    """
    This is the policy that we will be updating with each gradient update, we rollout using this policy's
    parameters, and we use the logprobs from this policy, we will also copy it's weights to make it old policy
    """
    
    def __init__(self, config):
        
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.config = config
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # first make a device mesh
        fsdp_size = int(int(os.environ['WORLD_SIZE']) / (config.ddp_size * config.tp_size))
        self.mesh = init_device_mesh(device,(config.ddp_size,fsdp_size, config.tp_size), mesh_dim_names=["DDP", "FSDP", "TP"])


    def prepare_optimizer(self):
        if self.config.tp_size > 1:
            self.model = prepare_tp_model(self.model, self.mesh)
        
        self.model = prepare_dp_model(self.model, self.mesh)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        # offload the model to cpu
        # load_model_to_device(self, "cpu")


#         if dist.get_rank() == 0:
#             print(f' after dp model rank: {dist.get_rank()} attention wq {self.model.layers[0].attention.wq.weight}')
        
class Actor(Worker):
    def __init__(self, config):
        super().__init__(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        #         define model from huggingface later on
        # simple_llama2_config = ModelArgs(dim=4, n_layers=1, n_heads=4, vocab_size=8)
        # self.model = Transformer.from_model_args(simple_llama2_config).to(device)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        # actor will need optimizer
        self.prepare_optimizer()


class Rollout(Worker):
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # init model using sglang
        self.prepare_env_var()

        if self.mesh["TP"].get_local_rank() == 0:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.engine = Engine(
                model_path=self.config.model_name,
                dtype="bfloat16",
                tp_size=self.mesh["TP"].size(),
                mem_fraction_static=0.5,
                # enable_memory_saver=True,
                port=30000 + dist.get_rank(),
            )

        # very important to do dist.barrier() i.e block the code right here, otherwise some gpu will go on.
        dist.barrier()


    async def rollout(self, data):

        messages,answer = data['messages'], data['answer']
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(ans)
        response = await self.engine.async_generate(
                prompt
            )
        # response = sample_response

        # generate reward
        reward = random.randint(0,1)

        messages.append({'role': 'assistant', 'content': response['text']})
        # Save list to a file
        # with open("messages.pkl", "wb") as f:
        #     pickle.dump(messages, f)

        # # print("List saved successfully!")

        # # Load list from the file
        # with open("messages.pkl", "rb") as f:
        #     messages = pickle.load(f)

        # let's find which tokens are states and which are actions
        # and make an action mask.

        states,actions, action_mask = [], [], []


        for message in messages:
            if message['role'] == 'assistant':
                state = self.tokenizer.encode(message['content'], add_special_tokens=False)
                action_mask.extend([1] * len(state))
                actions.extend(state)
            else:
                #else if it's a question/prompt
                prompt = self.tokenizer.apply_chat_template(message['content'], add_generation_prompt=True,tokenize=False)
                state = self.tokenizer.encode(prompt, add_special_tokens=False)
                action_mask.extend([0] * len(state))
                actions.extend([0]* len(state))

            states.extend(state)

        # sparse reward, only provide to the last token
        rewards = (len(states) - 1)*[0] + [reward]

        ex = {
            'states' : torch.LongTensor(states),
            'action_mask' : torch.LongTensor(action_mask),
            'rewards' : torch.FloatTensor(rewards)
        }

        return ex, messages

    def __call__(self, data_list):

        if self.mesh['TP'].get_local_rank() ==0:
            # There's still a bug here, some tp groups have the same data.
            data_list = split_data_list(data_list, mesh=self.mesh['DDP'])

            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                asyncio.gather(*(self.rollout(data) for data in data_list))
            )

            # later do this only when training
            self.engine.release_memory_occupation()
        dist.barrier()

        if self.mesh['TP'].get_local_rank() == 0:
            data_list, all_messages = map(list,zip(*outputs))

            print(f'rank {dist.get_rank()} gglen {len(data_list)}')

            # gather all the data_list 
            data_list = gather_data_list(data_list, self.mesh['DDP'])

        if dist.get_rank() == 0:
            return data_list


    def prepare_env_var(self):
        if (
            "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys()
        ):  # remove the use of common store for communication
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()

        # THE reason for doing this is because, we'll store rollout worker's (sglang) weight in these TP group
        # otherwise, SGL will use all the available cuda devices.
        cuda_visible_devices = [None] * self.config.tp_size
        dist.all_gather_object(
            cuda_visible_devices, os.environ["LOCAL_RANK"], self.mesh["TP"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)


import math

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
    
    return sum(lists, []) if rank==0 else None # if not group destination, lists wil be None, won't sum



class Trainer:
    def __init__(self, config):
        self.config = config
        check_mem_allocated(dist.get_rank(), 'before actor creation')
        self.actor = Actor(config)

        check_mem_allocated(dist.get_rank(), 'before rollout')

        # ------ turn it back on when needed ------
        self.rollout = Rollout(config)
        #  ------ turn it back on when needed ------

    def train(self):
        train_data = RLDataset(self.config.data_path, self.config.responses_per_prompt)
        train_dataloader = StatefulDataLoader(train_data, batch_size=self.config.per_rollout_size, drop_last=True, collate_fn=train_data.collate_fn)
        # construct train dataloader
        
        for data_list in train_dataloader:
            # print(f'rank {dist.get_rank()} lenght of data_list {len(data_list)}')
            # let's do the rollout --- turn it back on when doing real rollout ----
            data_list = self.rollout(data_list) # rank 0 will only have data_list, otherwise it'll be None
            # let's do the rollout --- turn it back on when doing real rollout ----


            check_mem_allocated(dist.get_rank(), 'after completing rollout')

            # save the data_list to picke so that
            if dist.get_rank == 0:

                with open('data_list.pkl', 'wb') as f:
                    pickle.dump(data_list, f)

            if dist.get_rank == 0:

                with open('data_list.pkl', 'rb') as f:
                    data_list = pickle.load(f)

            print(f'trn loop rank {dist.get_rank()} data_list length {len(data_list) if isinstance(data_list, list) else None}  \n\n' )

            break
            # generate rollouts. each train_batch will have length per_rollout_size x responses_per_prompt
            # first scatter the data across each ddp group




def check_mem_allocated(rank, msg):
    ans = torch.cuda.memory_allocated() / (1024**3)
    print(f'RANK {rank} MEMORY_ALLOCATED {msg} {ans}')


def start():
    # nest_asyncio.apply()
    config = Config()
    ppo_trainer = Trainer(config)
    ppo_trainer.train()

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