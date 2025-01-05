from torch import nn
import torch

import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_embd = config.n_embd
        self.n_head = config.n_head
    def forward(self,x):
        B,T,C = x.shape
        qkv = self.c_attn(x)
        
        Q,K,V = qkv.split(self.n_embd, dim=2)

        
        # comment out if using multi head attention
        ### ------ multi-head ----------------
        assert self.n_embd // self.n_head, 'n_embd and n_head dont comply'
        h_dim = self.n_embd // self.n_head
        
        Q = Q.view(B, T, self.n_head, C // self.n_head)
        K = K.view(B, T, self.n_head, C // self.n_head)
        V = V.view(B, T, self.n_head, C // self.n_head)
        Q = torch.transpose(Q, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        K = torch.transpose(K, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        V = torch.transpose(V, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        
        ### ------ multi-head ----------------
#         aw = (Q @ torch.transpose(K, -2,-1) * (1.0 / math.sqrt(K.size(-1)))) # for matmul dim of q should be B,T,C and k should be B,C,T
        
        aw = (Q @ torch.transpose(K, -2,-1)) # for matmul dim of q should be B,T,C and k should be B,C,T
        aw = aw/(K.shape[-1] **0.5)
        mask = self.tril[:,:,:T,:T] == 0 # generate mask
        aw = aw.masked_fill_(mask, float('-inf')) # apply mask i.e fill true values with -inf 
        aw = torch.softmax(aw,dim=-1) # -inf values are converted to 0 and then each row is normalized

        cv = aw @ V # context vector
        cv = torch.transpose(cv, 1,2) # bring it back to (B,T,n_heads, h_dim)
        cv = cv.contiguous().view(B,T,C)
        cv = self.c_proj(cv)
        return cv
        

        
        
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
    
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn = Head(config)
        self.mlp = FFN(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
    def forward(self,x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        
        return x

    
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_embd : int = 768
    n_head : int= 12
    block_size :int = 1024
    

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte' : nn.Embedding(self.config.vocab_size,self.config.n_embd),
            'wpe' : nn.Embedding(self.config.block_size, self.config.n_embd ),
            'h' : nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            'ln_f' : nn.LayerNorm(self.config.n_embd)
        })
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, x, targets=None):
        B, T = x.size()
        loss = None

        x_embd = self.transformer.wte(x)
#         pos = torch.arange(x.shape[1])
        pos = torch.arange(0, T, dtype=torch.long)
    
        x_pe = self.transformer.wpe(pos)
        
        x = x_embd + x_pe
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits,targets)

        return logits
        
        
    @classmethod
    def from_pretrained(cls,model_type):
        config = GPTConfig()
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = model.state_dict().keys()
        
        sd_keys = [key for key in sd_keys if not key.endswith('.attn.tril')]
        
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    
        sd_hf = model_hf.state_dict()
        sd_hf_keys = model_hf.state_dict().keys()
        
        assert len(sd_keys) == len(sd_hf_keys), 'keys unmatched, please make sure the length of keys matches'
        
        transposed = ['.attn.c_attn.weight', '.attn.c_proj.weight','.mlp.c_fc.weight', '.mlp.c_proj.weight', ]
        for k,v in sd_hf.items():
            for i in transposed:
                if k.endswith(i):
                    sd_hf[k] = sd_hf[k].t()
            with torch.no_grad():
#                 print(k)
#                 print(sd[k].shape, sd_hf[k].shape)
                sd[k].copy_(sd_hf[k])

                    
        return model
    
    
# -----------------------------------------------------------------------------
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
# model.to('cuda')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("My name is cutie siza")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
