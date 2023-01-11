import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data: pd.Series) -> None:
        super().__init__()
        self.X = data
        self.Y = [1] * len(self.X)  # existing interaction

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> object:

        x = self.X.iloc[index]  # item interactions of self.X at index
        y = self.Y[index]  # only 1
        return {"x": x, "y": y}
