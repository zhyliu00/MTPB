import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SGC(nn.Module):
    def __init__(self,nfeat):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nfeat, bias=False)
        self.activation = nn.Tanh()
        
    def forward(self, x, A):
        if(len(A.shape)==2):
            H = torch.einsum('ij,bjk->bik', A, x)
        else:
            H = torch.einsum('bij,bjk->bik', A, x)
        # H = torch.einsum('ij,bjk->bik', A, x)
        H = self.W(H)
        H = self.activation(H)
        return H