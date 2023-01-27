import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image

from pandas import DataFrame
from typing import Dict, Union


class MCNDataset(Dataset):
    def __init__(self, inter_data: DataFrame) -> None:
        super().__init__()
        self.inter_data = inter_data
        self.y = [1] * len(inter_data)

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index) -> Dict[str, Union[torch.Tensor, int]]:
        images = []

        for image_path in self.inter_data.iloc[index]:
            image = Image.open(image_path).convert("RGB")
            image = self.transform()(image)
            images.append(image)

        images = torch.stack(images)

        return {"x": images, "y": self.y[index]}

    def transform(self) -> torchvision.transforms.Compose:
        img_size = 224
        trans = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(img_size, img_size),
                torchvision.transforms.ToTensor(),
            ]
        )
        return trans
