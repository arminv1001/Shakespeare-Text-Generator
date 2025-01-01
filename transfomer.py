from torch.nn import functional as F
from torch import nn
import torch
import math
from self_attention import Self_Attention

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size)) # init with zeros
        pos_embedding[:, 0::2] = torch.sin(pos * den)   # sine waves
        pos_embedding[:, 1::2] = torch.cos(pos * den)   # cosine waves
        pos_embedding = pos_embedding.unsqueeze(-2)
        # no gradient to positional encodings
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, emd_dim, num_heads,dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.emd_dim = emd_dim
        self.linear_output = nn.Linear(num_heads*d_v, emd_dim)
        self.dropout = nn.Dropout(dropout)
        self.attentions = [Self_Attention(d_k, d_v, emd_dim) for _ in range(num_heads)]
    
    def forward(self, X):
        out = torch.cat([h(X) for h in self.attentions], dim=-1)
        out = self.dropout(self.linear_output(out))
        return out



class TransformerDecoder(nn.Module):
    def __init__(self, num_heads,d_k, d_v,emd_dim,vocab_size,output_layer = False, input_layer = False,dropout=0.2):
        super(TransformerDecoder, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        
        self.embd = nn.Embedding(vocab_size, emd_dim)
        self.positional_enc = PositionalEncoding(emd_dim)
        self.multi_head_attention = MultiHeadAttention(d_k, d_v, emd_dim, num_heads)
        self.laynorm1 = nn.LayerNorm(emd_dim)
        self.laynorm2 = nn.LayerNorm(emd_dim)
        self.laynorm3 = nn.LayerNorm(emd_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(emd_dim, 4*emd_dim),
            nn.ReLU(),
            nn.Linear(4*emd_dim, emd_dim),
            nn.Dropout(dropout)
        )

        self.linear_out = nn.Linear(emd_dim, vocab_size)
        
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    
    def forward(self, X):
        if self.input_layer:
            X = self.embd(X)
            X = self.positional_enc(X)
        
        X_residual = X
        X = self.multi_head_attention(self.laynorm1(X))
        X += X_residual
        
        X_residual = X
        X = self.feedforward(self.laynorm2(X))
        X += X_residual
        
        if self.output_layer:
            X = self.laynorm3(X)
            X = self.linear_out(X)
        
        return X
    
    def generate():
        pass
    
if __name__ == "__main__":
    vocab_size = 65
    emd_dim = 128
    transformer = TransformerDecoder(num_heads=8,d_k=64,d_v=64,emd_dim=emd_dim,vocab_size=vocab_size,output_layer=True, input_layer=True)
    print(transformer(torch.randint(0,1,(1,10))).shape)
    #print(transformer.generate(torch.randint(0,1,(1,64))))