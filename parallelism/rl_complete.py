

import os
import random
import torch
import asyncio
import pickle
import time
import gc
from qwen_monkey_patch import apply_qwen_patches
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
        # print('yass')
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
        # this mesh will only be used for model partition
        self.mesh = init_device_mesh(device,(config.ddp_size,fsdp_size, config.tp_size), mesh_dim_names=["DDP", "FSDP", "TP"])
        self.dp_size = int(int(os.environ['WORLD_SIZE']) / self.config.tp_size)

        # this mesh will be used for data parallelism 
        self.device_mesh = init_device_mesh(device,(self.dp_size, config.tp_size), mesh_dim_names=["DP", "TP"])

    def prepare_optimizer(self):

        self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            self.model = prepare_tp_model(self.model, self.mesh)
        
        self.model = prepare_dp_model(self.model, self.mesh)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        # Enable gradient checkpointing

        # offload the model to cpu
        load_model_to_device(self, "cpu")


#         if dist.get_rank() == 0:
#             print(f' after dp model rank: {dist.get_rank()} attention wq {self.model.layers[0].attention.wq.weight}')
        


def broadcast_data_list(data_list, mesh):

    # First get the length right across the same tp group
    if mesh.get_local_rank() == 0:
        len_data_list = torch.tensor(len(data_list)).to('cuda')
    else:
        len_data_list = torch.tensor(0).to('cuda')
    
    dist.broadcast(len_data_list, group=mesh.get_group(), group_src=0)
    
#     print(len_data_list.item())

    # then broadcast the same data_list across same tp group
    if mesh.get_local_rank() != 0:
        data_list = [None for _ in range(len_data_list)]
    
    dist.broadcast_object_list(data_list, group=mesh.get_group(), group_src=0)

    return data_list



class Rollout(Worker):
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.prepare_device_mesh()
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

    def prepare_device_mesh(self):
         
        dp_size = int(int(os.environ['WORLD_SIZE']) / self.config.tp_size)
        self.device_mesh = init_device_mesh("cuda", (dp_size, self.config.tp_size), mesh_dim_names=["DP", "TP"]) # device is on cpu cause we only need this mesh to scatter data (i.e for data parallelism)
        
    async def rollout(self, data):

        messages,answer = data['messages'], data['answer']

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        states = self.tokenizer.encode(prompt)
        actions = [0] * len(states)
        action_mask = [0] * len(states)

        # print(ans)
        response = await self.engine.async_generate(
                prompt, sampling_params={"temperature": self.config.temperature}
            )
        # response = sample_response

        # generate sparse reward
        reward = random.randint(0,1)
        messages.append({'role': 'assistant', 'content': response['text']})

        tokenized_response = self.tokenizer.encode(response['text'])
        states.extend(tokenized_response)
        actions.extend(tokenized_response)
        action_mask.extend([1] * len(tokenized_response))

        # sparse reward, only provide to the last token, putting extra -1 here cause later we do states[:-1]
        rewards = (len(states) -1 - 1)*[0] + [reward]


        ex = {
            'states' : torch.LongTensor(states[:-1]),
            'action_mask' : torch.LongTensor(action_mask[1:]),
            'rewards' : torch.FloatTensor(rewards),
            'actions' : torch.LongTensor(actions[1:])
        }

        return ex, messages

    def __call__(self, data_list):

        if self.device_mesh['TP'].get_local_rank() ==0:
            data_list = split_data_list(data_list, mesh=self.device_mesh['DP'])

            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                asyncio.gather(*(self.rollout(data) for data in data_list))
            )

            # later do this only when training
            self.engine.release_memory_occupation()
        dist.barrier()

        if self.device_mesh['TP'].get_local_rank() == 0:
            data_list, all_messages = map(list,zip(*outputs))

            # gather all the data_list 
            data_list = gather_data_list(data_list, self.device_mesh['DP'])
            # all_messages = gather_data_list(all_messages, self.device_mesh['DP'])

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
            cuda_visible_devices, os.environ["LOCAL_RANK"], self.device_mesh["TP"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def update(self, model):
        # first offload the model to cpu
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        state_dict = get_model_state_dict(
            model, options=options
        )

        # resume sglang's memory occupation
        torch.cuda.empty_cache()
        if self.device_mesh["TP"].get_local_rank() == 0:
            self.engine.resume_memory_occupation()
        
        for idx, (name, tensor) in enumerate(state_dict.items()):
            # load to gpu again, but this is a small tensor so it won't make much difference
            tensor = tensor.to(torch.cuda.current_device())
            # print(name)
            # if name == 'model.layers.0.self_attn.q_proj.weight':
                
            serialized_tensor = MultiprocessingSerializer.serialize(tensor.full_tensor() if isinstance(tensor, DTensor) else tensor)
            serialized_tensors = [None] * self.device_mesh['TP'].size() if self.device_mesh['TP'].get_local_rank() == 0 else None
            
            dist.gather_object(serialized_tensor, serialized_tensors, group_dst=0, group=self.device_mesh['TP'].get_group())
            
            if self.device_mesh["TP"].get_local_rank() == 0:
                # print(serialized_tensors)
                self.engine.update_weights_from_tensor(named_tensors=[(name, LocalSerializedTensor(values=serialized_tensors))])

            
            # print(f"rank {dist.get_rank()} seriliazed_tensor {serialized_tensor.shape} len_ST: {len(serialized_tensors) if isinstance(serialized_tensors,list) else serialized_tensors} ")

        dist.barrier()


        

def calc_logsumexp(tensor, mesh):

  step_size = 1024
  logsumexps = []
  for i in range(0,tensor.shape[1], step_size):
    logsumexps.append(torch.logsumexp(tensor[:,i:i+step_size,:], dim=-1))

  logsumexp = torch.cat(logsumexps, dim=-1)

  logsumexps = [torch.zeros_like(logsumexp) for _ in range(mesh['TP'].size())]

  dist.all_gather(logsumexps, logsumexp, mesh['TP'].get_group())

  ## ----- uncomment when using GPU ----------
  # logsumexps[device_mesh.get_local_rank()] = logsumexp # necessary to retain grad

  logsumexps = torch.stack(logsumexps, dim=-1)
  logsumexps = torch.logsumexp(logsumexps, dim=-1)


  return logsumexps
 
def differentiable_all_reduce(tensor, device_mesh):

    detached_tensor = tensor.detach()
    dist.all_reduce(
        detached_tensor,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return tensor + detached_tensor - tensor.detach()

def get_output_logits(logits, actions, mesh):
  # logits must be 1,T,C actions must be 1,T

  # each process will get its own logits shard.
  # actions is the same.

  # first we need to find which action belongs to which ranks.
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  local_vocab_size = torch.LongTensor([logits.shape[-1]]).to(device)

  gathered_vocab_sizes = [torch.zeros_like(local_vocab_size) for _ in range(mesh['TP'].size())]


  print(local_vocab_size.dtype)
  dist.all_gather(gathered_vocab_sizes, local_vocab_size, mesh['TP'].get_group())

  cu_vocab_size = torch.cumsum(
      torch.cat([torch.zeros_like(local_vocab_size)] + gathered_vocab_sizes), 0

  )
  # print(cu_vocab_size)

  action_device_mapping = (actions < cu_vocab_size[1:].unsqueeze(dim=-1)).to(torch.float32).argmax(dim=0) # dimension -> 1, no_of_seq
  # print(action_device_mapping)

  # get rank's actions.
  # now get which sequences belong to this rank
  rank = mesh['TP'].get_local_rank()

  # get the indices of non-zero elements
  local_action_indices = torch.nonzero(action_device_mapping == rank, as_tuple=True)[0]
  # print((local_action_indices))
  local_actions = actions[:, local_action_indices] - cu_vocab_size[rank]

  # logits is B,T,C. this T dimension is shared along all the local ranks, get only the logits for loca_action_indices
  local_logits = logits[:, local_action_indices]

  action_logits = torch.zeros(actions.shape, device=torch.cuda.current_device()).type_as(local_logits)
  print('action logits dtype', action_logits.dtype)
  action_logits[:,local_action_indices] = torch.gather(local_logits, -1, local_actions.unsqueeze(-1)).squeeze(-1)

  # now this action_logits needs to be all reduced.

  # return action_logits
  return differentiable_all_reduce(action_logits, device_mesh=mesh['TP'])



class Trainer:
    def __init__(self, config):
        self.config = config
        check_mem_allocated(dist.get_rank(), 'before actor creation')
        self.actor = Actor(config)

        check_mem_allocated(dist.get_rank(), 'after actor creation')

        # ------ turn it back on when needed ------
        # self.rollout = Rollout(config)
        #  ------ turn it back on when needed ------

    def train(self):
        train_data = RLDataset(self.config.data_path, self.config.responses_per_prompt)
        train_dataloader = StatefulDataLoader(train_data, batch_size=self.config.per_rollout_size, drop_last=True, collate_fn=train_data.collate_fn)
        # construct train dataloader
        
        for train_idx, data_list in enumerate(train_dataloader):
            if dist.get_rank() == 0:
                print(f' ----------------- TRAIN IDX {train_idx} ------------------') 

            # print(f'rank {dist.get_rank()} lenght of data_list {len(data_list)}')
            # let's do the rollout --- turn it back on when doing real rollout ----
            # data_list = self.rollout(data_list) # rank 0 will only have data_list, otherwise it'll be None
            # let's do the rollout --- turn it back on when doing real rollout ----

            check_mem_allocated(dist.get_rank(), 'after completing rollout')

            # save the data_list to picke so that
            # if dist.get_rank() == 0:

            #     with open('data_list.pkl', 'wb') as f:
            #         pickle.dump(data_list, f)

            ## --- simulate rollout ---
            data_list = None 
            if dist.get_rank() == 0:
                 
                with open('data_list.pkl', 'rb') as f:
                    data_list = pickle.load(f)

            # --- simulate rollout ---
            # print(f'trn loop rank {dist.get_rank()} data_list length {len(data_list) if isinstance(data_list, list) else None}  \n\n' )

            # ------ calculate the advantage ------
            if dist.get_rank() == 0:
                data_list = grpo_advantage(data_list, self.config.responses_per_prompt)
            # ------ calculate the advantage ------

            ## ---- old logprobs section ------
            # we've done the rollout, now let's generate the logprobs

            # since global rank 0 has the data, pass it to its other dp group members
            if self.actor.device_mesh['TP'].get_local_rank() == 0:
                data_list = split_data_list(data_list, self.actor.device_mesh['DP'])
            
            # for tp groups data must be same, so make same data
            data_list = broadcast_data_list(data_list, self.actor.device_mesh['TP'])
            
            with torch.no_grad():
                data_list = self.actor.compute_logprobs(data_list, log_type='old')

            break
            # collect the data_list from all dp groups, each dp group src 0 will get whole data, if we need to later divide the data, then why gather here, pointless right now
            # data_list = gather_data_list(data_list, self.actor.device_mesh['DP'])

            ## ----- old logprobs section -----
            # divide the data_list into minibatches, each of size, i.e total_rollout_data_in_this_rank / updates_per_rollout
            mini_batch_size = len(data_list) // self.config.updates_per_rollout

            data_list = [data_list[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(self.config.updates_per_rollout)]
            
            for update_step, minibatch in enumerate(data_list):
                # print(f' RANK {dist.get_rank()}-------- STEP: {update_step} ------------')
                minibatch_data_list = self.actor.compute_logprobs(minibatch, log_type="current")

                with open('minibatch.pkl', 'wb') as f:
                    pickle.dump(minibatch_data_list, f)
                # break
                

                
                # we need to multiply by dp_size.
                # Explanation, in standard pre-training, where each gpu processes some part of the batch_size, the gradients are averaged automatically so that the 
                # gradients would match if they were trained on one single machine
                # Ex: if we have batch_size= 32, with 2 gpus, then if we were training on single gpu, we would do loss/total_batch_size,
                # but if we are doing it on 2 gpus, each gpu will do loss/local_batch_size (i.e 16), that's why we would do, loss/16/2 = loss/32.
                # see how averaging is only done when sequences belong to the same batch, here in this RL step they do not, so we need to cancel out the auto-averaging.
                # thus the multiplication by self.dp_size
                loss = grpo_loss(minibatch, max_eps=self.config.max_eps, min_eps=self.config.min_eps) * self.actor.dp_size
                print(f'RANK {dist.get_rank()}-------- STEP: {update_step} ------------ loss: {loss} len_data_list {len(data_list)} ')
                loss.backward() # when we do this, the gradients are averaged among dp groups.
                
                check_mem_allocated(dist.get_rank(), 'before optimizer update')

                load_model_to_device(self.actor, torch.cuda.current_device())
                load_optimizer_to_device(self.actor, torch.cuda.current_device())

                check_mem_allocated(dist.get_rank(), 'after moving optimizer to GPU')

                self.actor.optimizer.step()
                self.actor.optimizer.zero_grad()
                load_optimizer_to_device(self.actor, "cpu")


                check_mem_allocated(dist.get_rank(), '---- clearing optimizer to cpu')
                # loss = compute_ppo_loss(minibatch_data_list)
                # then find the ppo loss.
                # do loss.backward()
                # then do optimizer.step()
                # self.actor.optimizer.step()

            ## ------- actor update section -------

            ## ------- actor update section -------

            # print(f'trn loop rank {dist.get_rank()} data_list length {len(data_list) if isinstance(data_list, list) else None} mesh {self.actor.device_mesh} datalist: {len(data_list[0])} \n\n' )
            # now lets do the update.


            # now the main ppo loss

            # generate rollouts. each train_batch will have length per_rollout_size x responses_per_prompt
            # first scatter the data across each ddp group
            # now let's update the slgang model with the trained model
            # takes in actor model, offload it to cpu, and piece by piece update the sglang model
            self.rollout.update(self.actor.model)



def grpo_advantage(data_list, responses_per_prompt):
  rewards = torch.tensor([ex['rewards'].sum() for ex in data_list]).view(-1, responses_per_prompt)
  baseline = rewards.mean(-1)
  std = rewards.std(-1)
  advantages = (rewards - baseline.unsqueeze(-1))/ (std.unsqueeze(-1) + torch.finfo(rewards.dtype).eps)
  # advantages = advantages.flatten()

  for ex, advantage in zip(data_list, advantages.flatten()):
    ex['advantage'] = advantage * ex['action_mask']

  return data_list


def check_mem_allocated(rank, msg):
    ans = torch.cuda.memory_allocated() / (1024**3)
    print(f'RANK {rank} MEMORY_ALLOCATED {msg} {ans}')


import torch

def grpo_loss(minibatch, max_eps, min_eps):
    max_len = max(item['old_logprobs'].shape[0] for item in minibatch)

    def pad_tensor(t, max_len):
        return torch.nn.functional.pad(t, (max_len - t.shape[0], 0))  # pad on left

    old_logprobs = torch.stack([pad_tensor(item['old_logprobs'], max_len) for item in minibatch], dim=0)
    logprobs     = torch.stack([pad_tensor(item['current_logprobs'], max_len) for item in minibatch], dim=0)
    advantage    = torch.stack([pad_tensor(item['advantage'], max_len) for item in minibatch], dim=0).to(torch.cuda.current_device())
    action_mask  = torch.stack([pad_tensor(item['action_mask'], max_len) for item in minibatch], dim=0).to(torch.cuda.current_device())
    
    ratio = torch.exp(logprobs - old_logprobs)
    ob1 = ratio * advantage
    ob2 = torch.clamp(ratio, 1.0 - min_eps, 1.0 + max_eps) * advantage

    ppo_loss = -torch.min(ob1, ob2) * action_mask

    loss = ppo_loss.sum(dim=-1)/ action_mask.sum(dim=-1) # mean across tokens

    return loss.mean() # mean across trajectories
    
    
    


class Actor(Worker):
    def __init__(self, config):
        super().__init__(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        #         define model from huggingface later on
        # simple_llama2_config = ModelArgs(dim=4, n_layers=1, n_heads=4, vocab_size=8)
        # self.model = Transformer.from_model_args(simple_llama2_config).to(device)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, attn_implementation="eager").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name) 
        # actor will need optimizer
        self.prepare_optimizer()

    # this func will only be used for old logprobs calculation so using torch.no_grad() 
    def compute_logprobs(self, data_list, log_type):
        # let's first split the data_list again across groups

        # print(f'RANK {dist.get_rank()} len data_list , {len(data_list) if isinstance(data_list, list) else None} first element {data_list[0] if isinstance(data_list, list) else None}')
        print(f'RANK {dist.get_rank()} len data_list , {len(data_list) if isinstance(data_list, list) else None} ')


        # recplicate the data across tp dimension cause they need the same data 
       
        # load the model back to gpu, previously the sharded model was stored in the CPU with it's reference contained in self.model
        load_model_to_device(self, torch.cuda.current_device())
        print('loaded model to device')

        input_ids = [item['states'] for item in data_list]
        action_input_ids = [item['actions'] for item in data_list]

        batch = {"input_ids": input_ids}
        action_batch = {"input_ids": action_input_ids}
        
        padded_input_ids = self.tokenizer.pad(batch, padding=True, padding_side='left') # make every row in the batch to have same length
        action_input_ids = self.tokenizer.pad(action_batch, padding=True, padding_side='left')['input_ids'].to('cuda')

        # print(f'rank {dist.get_rank()} and  padded input ids shape {padded_input_ids['input_ids'].shape} attention mask shape {padded_input_ids['attention_mask'].shape} ')
        padded_input_ids['input_ids'] = padded_input_ids['input_ids'].to('cuda')
        attention_mask = padded_input_ids['attention_mask'].to('cuda')
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0).to('cuda')

        if dist.get_rank() == 0:
            print(position_ids, padded_input_ids['input_ids'])    
        # print(f'rank {dist.get_rank()} and  padded input ids shape {padded_input_ids['input_ids'].shape} attention mask shape {padded_input_ids['attention_mask'].shape} position ids shape, {position_ids.shape}\n\n ')
        # print(f'rank {dist.get_rank()} position ids shape, ')


        # test_input_ids = torch.randint(0,2000, (2,100))
        # attention_mask = torch.ones_like(test_input_ids)
        # position_ids = attention_mask.long().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)
        # logits = self.model(input_ids=test_input_ids,attention_mask=attention_mask, position_ids=position_ids , use_cache=False).logits
        # print('this is logits shape', logits.shape)

        logits = self.model(input_ids=padded_input_ids['input_ids'], attention_mask=attention_mask, position_ids=position_ids).logits
        dist.barrier()

        print(f' rank {dist.get_rank()} logits {logits.shape}')
        print(f' rank {dist.get_rank()} logits {logits[-1][0][:5]}')

        # reconstruct action mask from padded_input_ids

        # if log_type=='old': # only update the action mask once, cause this function will be executed for calculating current logprobs
        #     action_mask = torch.zeros_like(padded_input_ids['input_ids']).to(torch.cuda.current_device())
        #     for idx, item in enumerate(data_list):
        #         len_actions = torch.sum(item['action_mask'])
        #         action_mask[idx, -len_actions:] = 1
        #         item['action_mask'] = action_mask[idx, :]
            
        # data_list['action_mask'] = action_mask
        # data_list['states'] = padded_input_ids

        B,T, vocab_size = logits.shape
        logsumexp = calc_logsumexp(logits, self.device_mesh)

        action_logits = get_output_logits(logits.view(1, B*T, vocab_size), action_input_ids.view(1,B*T), self.device_mesh).view(B,T)

        logprobs = action_logits - logsumexp

        for idx in range(logprobs.shape[0]):
            # get the logprobs for only the right side of logprobs that is equals to the actions length, cause left side has been padded to match max length
            data_list[idx][f'{log_type}_logprobs'] = logprobs[idx, -len(data_list[idx]['actions']):] # now oldlogprobs and actions will have the same length as actions and action_mask
            print( len(data_list[idx][f'{log_type}_logprobs']) == len(data_list[idx]['actions']))
        # data_list['old_logprobs'] = logprobs * data_list['action_mask']

        if dist.get_rank() == 0:
            print(f' actual logprobs {data_list[0]['old_logprobs']} ')
        # remove the cache
        # check_mem_allocated(dist.get_rank(), 'before clearing cache')
        # load_model_to_device(self, "cpu")
        del logits
        gc.collect()
        torch.cuda.empty_cache() 

        # check_mem_allocated(dist.get_rank(), 'after clearing cache')
        return data_list
        # now find the respective logprobs


def load_optimizer_to_device(worker, device):

    # if not getattr(worker.config, "offload_optimizer", False):
    #     return

    for param_group in worker.optimizer.param_groups:
        for param in param_group["params"]:
            state = worker.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(
                        device, non_blocking=True
                    )


def start():
    # nest_asyncio.apply()
    config = Config()
    apply_qwen_patches()
    ppo_trainer = Trainer(config)
    ppo_trainer.train()

    return

@dataclass
class Config:
    temperature: float = 1.0
    train_batch_size: int = 64
    model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    ddp_size: int = 1 
    tp_size: int = 2 
    lr: float = 1e-6
    data_path: str = 'CohleM/olympiad_small'
    responses_per_prompt: int = 2 
    per_rollout_size: int = 3
    offload_model: bool = True
    updates_per_rollout: int = 3
    max_eps: float = 0.02
    min_eps: float = 0.02
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