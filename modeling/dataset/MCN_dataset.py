import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image

from pandas import DataFrame
from typing import Tuple, List


class MCNDataset(Dataset):
    def __init__(self, inter_data: DataFrame, item_data: DataFrame) -> None:
        super().__init__()
        self.inter_data = inter_data
        self.item_data = item_data

        self.item_images = self.get_image_tensors()
        self.y = True

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, bool]:
        images = []

        for item_index in self.inter_data.iloc[index]:
            images.append(self.item_images[item_index])

        # shape (7, 3, img_size, img_size)
        # 7은 7개의 착용 장비정보, 3은 색깔 차원 RGB
        # 내 생각에는 (7, img_size, img_size, 3) 이 나와야 할 것 같은데
        # 일단 논문 코드에는 변형하는 부분이 없어서 그대로 진행
        images = torch.stack(images)

        return images, self.y

    def get_image_tensors(self) -> List[torch.Tensor]:
        """
        모든 아이템 이미지를 미리 텐서로 변환
        """
        image_tensors = [None for _ in range(len(self.item_data))]

        # 간혹 이미지가 오류가 나는 친구들도 존재
        # 그 경우 그냥 해당 카테고리의 평균 이미지 (dummy) 로 처리
        dummy = self.item_data[self.item_data["category"] == "dummy"]

        trans = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )

        for i, row in self.item_data.iterrows():
            image_path = row["local_image_path"]
            item_category = row["equip_category"]
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                image_path = dummy[dummy["equip_category"] == item_category][
                    "local_image_path"
                ].values[0]
                image = Image.open(image_path).convert("RGB")

            image_tensors[i] = trans(image)

        return image_tensors
