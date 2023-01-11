import torch
import pandas as pd
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data: pd.Series) -> None:
        super().__init__()
        self.X = data
        self.Y = [1] * len(self.X) # 조합이 있다는 의미
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, index) -> object:
        # print(index)
        # print(self.X.iloc[index])
        # print(self.Y[index])

        x = self.X.iloc[index] # self.X에서 index에 따른 아이템 배열을 뽑음
        y = self.Y[index] # y는 그냥 1
        return {"x": x, "y": y}
