import torch
import torch.nn.functional as F
from torch import nn

D = 256  # internal dimension
H = 4  # heads
N = 32768  # neurons
L = 6  # layers
dropout = 0.05
vocab_size = 256


class LinearAttention(nn.Module):
    def forward(Q, K, V):
        Qr = RoPE(Q)
        Kr = RoPE(K)
        return (Qr @ Kr.mT).tril(diagonal=-1) @ V


class BDH_GPU(nn.Module):
    def __init__(self):
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.wte = nn.Embedding(vocab_size, D)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Parameter(torch.zeros((N, D)).normal_(std=0.02))
        self.decoder_x = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.decoder_y = nn.Parameter(torch.zeros((H, D, N // H)).normal_(std=0.02))
        self.readout = nn.Parameter(torch.zeros((D, vocab_size)).normal_(std=0.02))
        self.attn = LinearAttention()

    def forward(self, idx):
        B, T = idx.size()  # mini−batchdimensions
        v_ast = self.ln(self.wte(idx).unsqueeze(1))  # B,1,T,D
        for i in range(L):
            x = F.relu(v_ast @ self.decoder_x)  # B,H,T,N//H
            a_ast = self.attn(
                Q=x,
                K=x,
                V=v_ast,
            )
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x  # B,H,T,N//H
            y = y.transpose(1, 2).reshape(B, 1, T, N)
            y = self.drop(y)
            # Start of layer with vectors x,y
            v_ast = v_ast + self.ln(y @ self.encoder)  # B,1,T,D
            v_ast = self.ln(v_ast)
        return v_ast.squeeze(1) @ self.readout  # B,T,vocab_size
