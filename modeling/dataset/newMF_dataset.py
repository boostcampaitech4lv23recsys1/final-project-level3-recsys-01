import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from typing import Dict, Union


class NewMFDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self.X = data
        self.Y = [1] * len(self.X)  # existing interaction

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Dict[str, Union[np.ndarray, int]]:

        x = self.X.iloc[index]  # item interactions of self.X at index
        y = self.Y[index]  # only 1
        return {"x": x.to_numpy(dtype=int), "y": y}
