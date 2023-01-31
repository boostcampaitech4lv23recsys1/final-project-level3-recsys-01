import torch
import torch.nn as nn
from typing import List


class NewMF(nn.Module):
    def __init__(self, n_items: int, n_factors: int) -> None:
        super().__init__()

        self.item_factors = nn.Embedding(n_items, n_factors, sparse=True)

        # self.item_factors.weight.data.uniform_(-1, 1) # uniform distribution weight init
        self.sigmoid = nn.Sigmoid()

    def forward(self, items: torch.Tensor) -> torch.Tensor:

        # 배치마다 학습에 사용하지 않는 부분이 달라서 그냥 일단 for loop 돌림
        # 나중에 조금 더 고민해봐야 할 것 같다. 혹은 MCN 모델에는 이미 구현이 되어 있겠지?
        """
        이렇게 하면 masking 된 부분을 1로 바꿀 수 있다.
        착용하지 않은 아이템을 -1로 해놨는데, 0으로 바꿔야 사용 가능.
        -1이면 self.item_factors(items) 할 때 음수가 들어가서 오류난다.
        mask = items != 0
        vector = self.item_factors(items)
        vector[~mask] = 1
        """
        inner_product = []

        for items_per_batch in items:
            items_per_batch = items_per_batch[items_per_batch != -1]
            item_emb = self.item_factors(items_per_batch)
            inner_product.append(torch.sum(torch.prod(item_emb, dim=0), dim=0))

        inner_product = torch.stack(inner_product, dim=0)

        return self.sigmoid(inner_product)