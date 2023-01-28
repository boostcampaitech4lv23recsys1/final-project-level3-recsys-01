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
        self.y = True

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index) -> Dict[str, Union[torch.Tensor, int]]:
        images = []

        for image_path in self.inter_data.iloc[index]:
            image = Image.open(image_path).convert("RGB")
            image = self.transform()(image)
            images.append(image)

        # shape (7, 3, img_size, img_size)
        # 7은 7개의 착용 장비정보, 3은 색깔 차원 RGB
        # 내 생각에는 (7, img_size, img_size, 3) 이 나와야 할 것 같은데
        # 일단 논문 코드에는 변형하는 부분이 없어서 그대로 진행
        images = torch.stack(images)

        return images, self.y

    def transform(self) -> torchvision.transforms.Compose:
        img_size = 224
        trans = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((img_size, img_size)),
                torchvision.transforms.ToTensor(),
            ]
        )
        return trans
