import pickle
import math
from collections import defaultdict
from datasets import load_dataset
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Download Dataset TODO TODO TODO Remove from here???
dataset = load_dataset("tatsu-lab/alpaca")

import mlx.core as mx

class mxUtils:
   
    @staticmethod
    def tril(matrix: mx.array) -> mx.array:
        """
        Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, 
        the other elements of the result array out are set to 0.
        """
        r, c = matrix.shape
        for i in range(r):
            for j in range(c):
                if i < j:
                    matrix[i, j] = 0
        return matrix
    
    @staticmethod
    def unsqueeze(matrix: mx.array, dim: int) -> mx.array:
        """
        Add a singleton dimension to an MLX array at the specified index.
        """
        new_shape = list(matrix.shape)
        new_shape.insert(dim, 1)
        return mx.reshape(matrix, new_shape)

class GPTConfig:
    
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1
    
    def __init__(
        self, vocab_size, max_len, **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class GPT1Config(GPTConfig):
    
    num_heads = 12
    num_blocks = 12
    embed_dim = 768
    
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        embed_dim = config.embed_dim
        self.max_len = config.max_len
        self.tok_embed = nn.Embedding(config.vocab_size, embed_dim)
        self.pos_embed = mx.zeros([1, self.max_len, embed_dim])
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_blocks)])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, config.vocab_size)
    
    def __call__(self, x, target=None):
        seq_len = x[1].shape[0]
        assert seq_len <= self.max_len, f"Sequence of length {seq_len} longer than max of {self.max_len}!"
        tok_embedding = self.tok_embed(x)
        pos_embedding = self.pos_embed[:, :seq_len, :]
        x = self.dropout(tok_embedding + pos_embedding)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )
    
    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, f"Invalid Heads and Embedding Dimension configuration! Heads:{self.num_heads} Embedding Dimensions: {self.embed_dim}"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.mask = mxUtils.unsqueeze(mxUtils.unsqueeze(mxUtils.tril(mx.ones([config.max_len, config.max_len])), 0), 0)
    
    def __call__(self, x):
        batch_size = x[0].shape[0]
        seq_len = x[1].shape[0]
        k_t = mx.transpose(self.key(x).reshape(batch_size, seq_len, self.num_heads, -1), axes=[0, 2, 3, 1])
        v = mx.transpose(mx.reshape(self.value(x), [batch_size, seq_len, self.num_heads, -1]), axes=[0, 2, 1, 3])
        q = mx.transpose(mx.reshape(self.query(x), [batch_size, seq_len, self.num_heads, -1]), axes=[0, 2, 1, 3])
        
        attn = mx.matmul(q, k_t) / math.sqrt(q[-1].shape[0])
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf")) # TODO START HERE: figure out how to replace this shit
        attn = self.attn_dropout(attn)
        attn = mx.softmax(attn, axis=-1)
        
        y = mx.matmul(attn, v)
        y = y.transpose(1, 2)
        y = y.reshape(batch_size, seq_len, -1)
        y = self.proj_dropout(self.proj(y))
        return y

# Using the GPT model!

vocab_size = 10
max_len = 12

config = GPT1Config(vocab_size, max_len)
model = GPT(config)

batch_size = 3
seq_len = 6

test_input = mx.random.randint(low=0, high=vocab_size, shape=[batch_size, seq_len])
try:
    print(model(test_input).shape)
except AssertionError as e:
    print(e)