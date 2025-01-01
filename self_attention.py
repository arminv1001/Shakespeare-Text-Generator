import torch
import torch.nn as nn
from torch.nn import functional as F

class Self_Attention(nn.Module):
    def __init__(self, d_k, d_v,emd_dim,dropout=0.2,block_size=10):
        super(Self_Attention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Linear(emd_dim, d_k,bias=False)
        self.W_k = nn.Linear(emd_dim, d_k,bias=False)
        self.W_v = nn.Linear(emd_dim, d_v,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, mask=True):
        
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        
        Q_K = torch.matmul(Q,K.transpose(-2,-1))
        Q_K_scaled = Q_K * (self.d_k**-0.5)
        if mask == True:
            Q_K_scaled = Q_K_scaled.masked_fill(self.tril[:X.shape[1], :X.shape[1]] == 0, float('-inf'))
        Q_K_softmax = F.softmax(Q_K_scaled, dim=-1)
        Q_K_softmax = self.dropout(Q_K_softmax)
        output = torch.matmul(Q_K_softmax, V)
        return output
    
    def generate(self, X, embd,max_gen_tokens=100):
        for _ in range(max_gen_tokens):
            idx_cond = X[:, -X.shape[1]:]
            idx_cond = embd(idx_cond)
            output = self(idx_cond)
            output = output[:, -1, :]
            probs = F.softmax(output, dim=-1) 
            X_next = torch.multinomial(probs, num_samples=1) 
            X = torch.cat((X, X_next), dim=1) 
        return X
    

if __name__ == "__main__":
    vocab_size = 100
    emd_dim = 128
    embedding = nn.Embedding(100,emd_dim)
    self_attention = Self_Attention(d_k=64,d_v=64,emd_dim=emd_dim)
    print(self_attention(embedding(torch.randint(0,1,(1,300)))))
    print(self_attention.generate(torch.randint(0,1,(1,64)),embedding))
        