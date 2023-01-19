import torch
import torch.nn as nn

class NewMF(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors) -> None:
        super().__init__()

        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

        self.item_factors.weight.data.uniform_(-1, 1) # uniform distribution weight init
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, items):
        item_emb = self.item_factors(items[0])
        for i in range(1, len(items)):
            item_emb = item_emb * self.item_factors(items[i])
            
        return self.sigmoid(item_emb.sum(-1))
    