# Big idea:
#   - We convert text -> token IDs
#   - We train a Transformer to predict the next token at every position
#   - We use a causal mask so a position cannot attend to future tokens
#   - We can then generate new text autoregressively

import math         
import torch              
import torch.nn as nn                
import torch.nn.functional as F      

# ----------------------------
# 1) Tiny "tokenizer" (character-level)
# ----------------------------

text = "hello world! my name is eeko and I am a pitbull mix. I love to play fetch and cuddle. nice to meet you! caleb is cool." # create a tiny corpus by repeating a short string many times
chars = sorted(list(set(text)))         # unique characters in the corpus, sorted for consistent ordering
vocab_size = len(chars)                 # number of unique characters (vocabulary size)

stoi = {ch: i for i, ch in enumerate(chars)}  # map each character -> integer ID
itos = {i: ch for ch, i in stoi.items()}      # inverse map: integer ID -> character

def encode(s):
    # convert a string into a 1D tensor of token IDs
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(ids):
    # convert a sequence of token IDs back into a Python string
    return "".join([itos[int(i)] for i in ids])

data = encode(text)                      # the entire corpus as a 1D tensor of IDs, shape (T,)

# ----------------------------
# 2) Make training batches (x -> y shifted by 1)
# ----------------------------

def get_batch(data, batch_size, block_size, device):
    # Choose random starting indices so each batch is different
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))  # (B,)

    # Build input sequences x: tokens [i, ..., i+block_size-1]
    x = torch.stack([data[i:i+block_size] for i in ix])               # (B, S)

    # Build target sequences y: the "next token" for each position
    # y is the same as x but shifted by 1 position in the original text
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])           # (B, S)

    # Move tensors to the chosen device (CPU/GPU)
    return x.to(device), y.to(device)

# ----------------------------
# 3) Causal attention mask
# ----------------------------

def causal_mask(seq_len, device):
    # Create an upper-triangular matrix of True values above the diagonal
    # True means "mask out" (disallow attention to that position)
    #
    # Example for seq_len=4:
    # [[False, True,  True,  True],
    #  [False, False, True,  True],
    #  [False, False, False, True],
    #  [False, False, False, False]]
    #
    # This ensures token at position i can only attend to <= i (no future tokens).
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

# ----------------------------
# 4) Tiny GPT-like language model (Transformer encoder blocks + causal mask)
# ----------------------------

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, block_size=64, dropout=0.1):
        super().__init__()                                   # initialize nn.Module internals

        self.vocab_size = vocab_size                         # store vocab size (number of possible tokens)
        self.d_model = d_model                               # store embedding dimension
        self.block_size = block_size                         # maximum sequence length this model supports

        self.tok_emb = nn.Embedding(vocab_size, d_model)     # token embedding lookup table: (vocab -> d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)     # positional embeddings: (position -> d_model)

        # Define a single Transformer encoder block
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                                 # embedding dimension
            nhead=nhead,                                     # number of attention heads
            dim_feedforward=4*d_model,                       # hidden size in the MLP inside the block
            dropout=dropout,                                 # dropout probability
            activation="gelu",                               # activation in the block MLP
            batch_first=True,                                # use tensor shape (B, S, E) instead of (S, B, E)
            norm_first=True                                  # pre-LayerNorm style (common in modern Transformers)
        )

        self.transformer = nn.TransformerEncoder(            # stack multiple encoder blocks
            enc_layer,                                       # the block definition
            num_layers=num_layers                            # how many blocks to stack
        )

        self.ln_f = nn.LayerNorm(d_model)                    # final LayerNorm before the output head
        self.head = nn.Linear(d_model, vocab_size)           # map hidden states -> logits over vocabulary

    def forward(self, idx):
        # idx: (B, S) tensor of token IDs
        B, S = idx.shape                                     # batch size and sequence length
        assert S <= self.block_size                          # prevent sequences longer than positional embedding table

        pos = torch.arange(S, device=idx.device)             # positions [0..S-1], shape (S,)
        tok = self.tok_emb(idx)                              # token embeddings, shape (B, S, E)
        pos = self.pos_emb(pos)                              # position embeddings, shape (S, E)
        x = tok + pos                                        # broadcast add positions to each batch, shape (B, S, E)

        attn_mask = causal_mask(S, idx.device)               # causal mask, shape (S, S)
        x = self.transformer(x, mask=attn_mask)              # apply stacked Transformer blocks, shape (B, S, E)

        x = self.ln_f(x)                                     # final normalization, shape (B, S, E)
        logits = self.head(x)                                # vocabulary logits, shape (B, S, V)
        return logits                                        # return unnormalized scores (logits)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0):
        # idx: (B, S) starting prompt tokens
        # max_new_tokens: how many tokens to append
        # temperature: >1 makes output more random, <1 makes output more confident/peaky
        self.eval()                                          # evaluation mode (disables dropout)

        for _ in range(max_new_tokens):                      # generate one token at a time
            idx_cond = idx[:, -self.block_size:]             # if context too long, keep only last block_size tokens

            logits = self(idx_cond)                          # forward pass, shape (B, S, V)
            next_logits = logits[:, -1, :]                   # take last position logits, shape (B, V)
            next_logits = next_logits / temperature          # apply temperature scaling

            probs = F.softmax(next_logits, dim=-1)           # convert logits -> probabilities, shape (B, V)
            next_id = torch.multinomial(probs, num_samples=1)# sample next token ID, shape (B, 1)

            idx = torch.cat([idx, next_id], dim=1)           # append sampled token to sequence, shape (B, S+1)

        return idx                                           # return full sequence (prompt + generated)

# ----------------------------
# 5) Training setup
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose GPU if available, else CPU
torch.manual_seed(0)                            # set random seed for reproducibility

block_size = 64                                 # context window length (sequence length)
model = TinyTransformerLM(                      # instantiate the model
    vocab_size=vocab_size,                      # vocabulary size from tokenizer
    d_model=128,                                # embedding dimension
    nhead=4,                                    # attention heads
    num_layers=2,                               # transformer layers
    block_size=block_size                       # max sequence length supported
).to(device)                                    # move model parameters to CPU/GPU

opt = torch.optim.AdamW(                        # AdamW optimizer (common for Transformers)
    model.parameters(),                         # optimize all model parameters
    lr=3e-4,                                    # learning rate
    weight_decay=1e-2                           # weight decay regularization
)

def lm_loss(logits, targets):
    # logits: (B, S, V) and targets: (B, S)
    B, S, V = logits.shape                      # unpack dimensions
    logits_2d = logits.reshape(B*S, V)          # flatten batch+time into one dimension
    targets_1d = targets.reshape(B*S)           # flatten targets similarly
    return F.cross_entropy(logits_2d, targets_1d)  # standard next-token cross entropy

steps = 500                                     # how many gradient updates
batch_size = 64                                 # how many sequences per batch

for step in range(1, steps + 1):                # main training loop
    model.train()                               # training mode (enables dropout)

    x, y = get_batch(                           # get a random batch
        data=data,                              # tokenized corpus
        batch_size=batch_size,                  # batch size
        block_size=block_size,                  # sequence length
        device=device                           # CPU/GPU
    )

    logits = model(x)                           # forward pass: predict logits for each position
    loss = lm_loss(logits, y)                   # compute next-token loss

    opt.zero_grad()                             # clear old gradients
    loss.backward()                             # backprop compute gradients
    torch.nn.utils.clip_grad_norm_(             # clip gradients to stabilize training
        model.parameters(),                     # params to clip
        max_norm=1.0                            # max gradient norm
    )
    opt.step()                                  # update parameters

    if step % 100 == 0:                         # print occasionally
        print(f"step {step:4d} | loss {loss.item():.4f}")

# ----------------------------
# 6) Text generation demo
# ----------------------------

prompt = "caleb "                               # starting text prompt
idx0 = encode(prompt).unsqueeze(0).to(device)   # encode prompt -> (S,) then add batch dim -> (1, S)

out_ids = model.generate(                       # generate continuation
    idx=idx0,                                   # starting tokens
    max_new_tokens=10,                         # how many tokens to generate
    temperature=1                            # sampling temperature
)[0].cpu()                                      # take first batch element and move to CPU

print("\n--- generated ---")
print(decode(out_ids))                          # decode token IDs back to text

