import torch
from torch.utils.data import Dataset

from itertools import product

from typing import List, Dict


class MCNTopkDataset(Dataset):
    def __init__(
        self, equips: List[int], part_index: int, item_part: List[int]
    ) -> None:
        super().__init__()
        self.equips = equips  # 현재 착용 장비 정보
        self.part_index = part_index  # 빈 곳. 바꿔야 할 아이템 자리
        self.item_part = item_part  # 바꿔야 할 아이템들의 모든 index 값

    def __len__(self) -> int:
        return len(self.item_part)

    def __getitem__(self, index) -> torch.Tensor:

        new_item = self.item_part[index]
        equips_copy = self.equips.copy()
        equips_copy[self.part_index] = new_item

        return torch.LongTensor(equips_copy)


class MCNProductDataset(Dataset):
    def __init__(
        self, equips: List[int], part_index_and_topk: Dict[int, List[int]]
    ) -> None:
        super().__init__()
        self.equips = equips  # 현재 착용 장비 정보
        # 빈 곳에 대한 인덱스가 키, topk개의 index가 리스트에 담김
        self.part_index_and_topk = part_index_and_topk

        # 모든 경우의 수 저장
        self.possible_product = list(product(*part_index_and_topk.values()))

    def __len__(self) -> int:
        return len(self.possible_product)

    def __getitem__(self, index: int) -> torch.Tensor:
        changed_part = self.possible_product[index]
        equips_copy = self.equips.copy()
        for i, part_index in enumerate(self.part_index_and_topk):
            equips_copy[part_index] = changed_part[i]
        return torch.LongTensor(equips_copy)
