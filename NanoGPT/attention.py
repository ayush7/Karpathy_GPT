"""
[Dataset] https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import torch 
import torch.nn as nn
from torch.nn import functional as F 

# Hyperparameters
batch_size = 32 
block_size = 8 
max_iter = 5000 
eval_interval = 500 
learning_rate = 1e-3 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ",device)
eval_iters = 200 
torch.manual_seed(1337)
n_embed = 32

with open('dataset/input.txt','r',encoding='utf-8') as f:
    text = f.read()

# Check unique characters 
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # Function takes a string and retuns encoded integers
decode = lambda l: ''.join([itos[i] for i in l]) # Function takes a list of integers and returns a decoded string


# Train and test splits

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    # generate a small batch of input data x and target y 
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y 

# xb, yb = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

"""
Self Attention Module

"""
class Head(nn.Module):
    """ One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)   # B,T,C
        q = self.query(x) # B,T,C

        # Compute attention score (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (BTC) @ (BCT) --> (BTT)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # BTT
        wei = F.softmax(wei, dim=-1)

        # Perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v 
        return out 
"""
Implement Multi Head Attention
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads) ])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        mha = torch.cat([h(x) for h in self.heads], dim=-1) 
        # Concat happens in the channels dim and n_heads are adjusted while calling to accomodate
        mha = self.proj(mha)
        return mha


class FeedForward(nn.Module):
    """Simple feed forward network with an activation"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed,n_embed)
        )

    def forward(self, x):
        return self.net(x)

"""
Making a decoder block for self attention
"""

class Block(nn.Module):
    """ Block : Communication followed by computation [thinking]"""
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed//n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ff = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x


"""
The GPT Language Model Class

"""
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4),
            nn.LayerNorm(n_embed),
            )
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape 

        # idx and targets are both B,T tensor of integers
        tok_emb = self.token_embedding_table(idx)                                # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb                                                    # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)                                           # (B,T,vocab_size)

        # logits = self.token_embedding_table(idx)  # Batch, Time(block size), Channel(vocab size) = BTC
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print(B, T, C)

            logits = logits.view(B*T , C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits , loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            # crop the idx to prevent scope errors
            idx_cond = idx[:, -block_size:]
            # get prediction
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # B,C
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next),dim=1) #(B,T+1)
        return idx 
    
model = GPTLanguageModel()
m = model.to(device)


# Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iter):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))