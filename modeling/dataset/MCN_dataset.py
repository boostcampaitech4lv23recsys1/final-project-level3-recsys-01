import torch
import torchvision
from torch.utils.data import Dataset

from PIL import Image
import random
import pandas as pd

from typing import Tuple, List
from tqdm import tqdm


class MCNDataset(Dataset):
    def __init__(
        self,
        inter_data: pd.DataFrame,
        item_data: pd.DataFrame,
        negative_ratio: float,
        n_change_parts: int,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.inter_data = inter_data
        self.item_data = item_data

        if not 0 <= negative_ratio <= 1:
            raise ValueError(
                f"negative_ratio는 0~1 사이의 실수 값 이어야 합니다. 현재 값: {negative_ratio}"
            )
        if not 1 <= n_change_parts <= 7:
            raise ValueError(
                f"n_change_parts는 1~7 사이의 정수 값 이어야 합니다. 현재 값: {n_change_parts}"
            )
        self.negative_ratio = negative_ratio
        self.n_change_parts = n_change_parts
        self.is_train = is_train

        # 카테고리 별 dataframe 미리 생성
        # 카테고리 별 negative sampling을 하기 위함.
        # Top이랑 Overall은 그냥 같이 묶어서 negative sampling 진행
        self.categories = ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]
        self.items_per_category = []
        for category in self.categories:
            if category == "Top":
                overall = self.item_data[self.item_data["equip_category"] == "Overall"]
                top = self.item_data[self.item_data["equip_category"] == category]
                item_per_category = pd.concat((top, overall)).reset_index(drop=True)
            else:
                item_per_category = self.item_data[
                    self.item_data["equip_category"] == category
                ].reset_index(drop=True)
            self.items_per_category.append(item_per_category)

        print("----------이미지를 텐서로 변환하기 시작합니다. ----------")
        self.item_images = self.get_image_tensors()

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, bool]:
        item_indexs = self.inter_data.iloc[index].tolist()

        # 주어진 비율보다 작은 값이 나오면
        negative = False
        if random.random() < self.negative_ratio and self.is_train:
            negative = True
            for change_part_index in random.sample(range(7), self.n_change_parts):
                item_per_category = (
                    self.items_per_category[change_part_index]
                    .sample(n=1, replace=False)
                    .iloc[0]["index"]
                )
                item_indexs[change_part_index] = item_per_category

        images = []

        for item_index in item_indexs:
            images.append(self.item_images[item_index])

        # shape (7, 3, img_size, img_size)
        # 7은 7개의 착용 장비정보, 3은 색깔 차원 RGB
        # 내 생각에는 (7, img_size, img_size, 3) 이 나와야 할 것 같은데
        # 일단 논문 코드에는 변형하는 부분이 없어서 그대로 진행
        images = torch.stack(images)

        return images, not negative

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

        for i, row in tqdm(self.item_data.iterrows()):
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
