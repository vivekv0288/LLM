import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import config
from model import GPTConfig
from model import GPT
from model import CausalSelfAttention
from model import MLP
from model import Block


def encode( word):
    stoi = {ch:i for i,ch in enumerate(config.vocab)}
    itos = {i:s for s,i in stoi.items()}
    ix = torch.tensor([stoi[w] for w in word], dtype=torch.long)
    return ix
def decode( ix):
    stoi = {ch:i for i,ch in enumerate(config.vocab)}
    itos = {i:s for s,i in stoi.items()}
    word = ''.join(itos[i] for i in ix)
    return word

ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
master_process = True
# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

checkpoint = torch.load(config.checkpoint_path)

# Initialize the model using the configuration saved in the checkpoint
model = GPT(checkpoint['config'])

# Load the saved state dict into the model
model.load_state_dict(checkpoint['model'])
model.to(device)
use_compile = False
# Optionally, retrieve the training step to continue from where it left off
trained_step = checkpoint['step']

model.eval()
num_return_sequences = config.num_return_sequences
random_number = random.randint(0, 25)
tokens =[52, random_number]
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42 + ddp_rank)
while xgen.size(1) < config.generate_max_length:
    # forward the model to get the logits
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(xgen) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
# print the generated text
for i in range(config.num_return_sequences):
    tokens = xgen[i, :config.generate_max_length].tolist()
    decoded = decode(tokens)
    print(f"rank {ddp_rank} sample {i}: {decoded}")