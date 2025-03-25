""" 

This script was done in order to learn and better grasp the concept of transformers. Mostly,
it is motivated by Andrej Karpathy's video: https://www.youtube.com/watch?v=kCc8FmEb1nY.
I might extend this a bit in the future and implement the encoder as well. Another possible direction I might pursue is
upgrading the architecture to work on a word-level basis, as currently it works on character-level. 

"""

import torch
import torch.nn as nn  
import math
from torch.nn import functional as F
from datasets import load_dataset

# set seed and device
torch.manual_seed(1337)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 16
block_size = 32 # sequence length
max_iters = 10000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# -----------

# here are all the unique characters that occur in this text
df = load_dataset("Teejeigh/raw_friends_series_transcript")['train']['text']
text = "\n".join(df)
chars = sorted((set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# split data
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class PositionalEncoding(nn.Module):
    """ positional encoding for transformer models """

    def __init__(self, n_embd, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(max_len, 1, n_embd)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # This is typically used to register a buffer that should not to be considered a model parameter.
    def forward(self, x):
        return self.pe[:x.size(0)]

class Head(nn.Module):
    """self-attention head"""
    def __init__(self, head_size = 16, n_emb = n_embd):
        super().__init__()
        self.queries = nn.Linear(n_emb, head_size, bias = False)
        self.keys = nn.Linear(n_emb, head_size, bias = False)
        self.values = nn.Linear(n_emb, head_size, bias = False) 
        self.dropout = nn.Dropout(0.1)
        self.embedding_dim = head_size

    def forward(self, x):
        B, T, C = x.shape

        q = self.queries(x).to(device) # B T 16
        k = self.keys(x).to(device) # B T 16 we want -> B 16 T
        mask = (q @ k.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        tril = torch.tril(torch.ones(T, T).to(device))
        mask = mask.masked_fill(tril == 0, float('-inf'))
        mask = F.softmax(mask, dim=-1)
        mask = self.dropout(mask)
        
        v = self.values(x)
        out = mask @ v
        return out

class MultiHead(nn.Module):
    """ multiple self-attention heads """
    def __init__(self, num_heads, head_size, n_emb = n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb) # NOTE: Why is this necessary? 
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
class MLP(nn.Module):
    """ a simple feed-forward neural network """
    def __init__(self, n_embd = n_embd):
        super().__init__()
        self.layer1 = nn.Linear(n_embd, n_embd * 4) # *4 can be changed
        self.layer2 = nn.Linear(4 * n_embd, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x): # x is the output from the self attention 
        x = self.relu(self.layer1(x))
        x = self.dropout(self.layer2(x))
        return x
    
class Block(nn.Module):
    """ a block in a transformer, consisting of multi-head self-attention, feed forward neural network and layernorm """
    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head 
        self.self_attn = MultiHead(n_head, head_size)
        self.mlp = MLP(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x)) # first x is residual connection
        x = x + self.mlp(self.ln2(x)) # same here
        return x 
    
class Decoder(nn.Module):
    """ decoder component from the "Attention is all you need" paper """
    # the same ones as encoder + third which performs:
    # multi-head attention over output of encoder (if seq-to-seq)
    # residual connectiions
    # mask future tokens ! -> only difference from encoder
    # output embeddings are offset by one position
    # prediction for position i must depend only on known outputs at that time i
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = PositionalEncoding(n_embd, block_size) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets = None):
        input_emb = self.embedding(x) # B T C 
        pos_emb = self.positional_embedding(input_emb)
        x = input_emb + pos_emb # B T C
        x = self.blocks(x) # B T C
        x = self.ln(x) # B T C
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """ used for next character (or word in the future) generation """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens - NOTE: Foggy?
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

class Encoder(nn.Module):
    # first sub-layer -> multi head self attention
    # second -> simple position wise fully connected feed-forward net
    # residual connection around sub layers
    # + layer normalization
    # this combined forms a Block
    pass
    # TODO: implement this 

def train_model(model, optimizer, max_iters, eval_interval):
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # get a batch of data
        xb, yb = get_batch('train')

        # evaluate
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def generate_text(model, max_new_tokens=500):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
    print("Generated text:")
    print(generated_text)

def main():
    # model
    model = Decoder()
    model = model.to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # train
    train_model(model, optimizer, max_iters, eval_interval)
    # generate
    generate_text(model)

if __name__ == "__main__":
    main()


# NOTE(s):
# In encoder-decoder attention, queries come from previous decoder layer,
# and keys values come from the output of the encoder. This allows every
# position in the decoder to attend over all positions in the input sequence. This mimics the
# typical encoder-decoder attention mechanisms in sequence-to-sequence models. e.g. translation task

# encoder only. Self-attention, all the keys values and queries come from 
# the same place, in this case, output from the previous layer in the encoder.
# each position in the encoder can attend to all positions.

# decoder. Similar, however with one difference: 
# We need to prevent leftward information flow in the decoder to 
# preserve the auto-regressive property. We implement this
# inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
# of the softmax which correspond to illegal connections. 