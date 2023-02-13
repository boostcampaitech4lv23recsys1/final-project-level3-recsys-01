from src.AI.models.model_object import AETopkDataset, AEProductDataset, AutoEncoderPredictor
from src.utils.gcs_helper import GCSHelper
from src.database.models.crud_item import find_by_item_idxs
from src.database.init_db import get_db

import torch
from torch.utils.data import DataLoader

import os
import random

from typing import Dict, List, Any
from pandas import DataFrame


class AEInference:
    def __init__(
        self,
        model_config: Dict[str, Any],
        image_tensors: torch.Tensor,
        dummy: DataFrame,
    ) -> None:
        self.model_config = model_config
        self.model_path = model_config["model_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(self.model_path):
            os.makedirs("/".join(self.model_path.split("/")[:-1]))
            gcs_helper = GCSHelper(
                key_path="src/utils/gcs_key.json", bucket_name="maple_trained_model"
            )
            gcs_helper.download_file_from_gcs(
                blob_name="AutoEncoderPredictor/AutoEncoderPredictor_latest.pt", file_name=self.model_path
            )

        self.item_parts = list()
        self.top_k = model_config["top_k"]
        self.batch_size = model_config["batch_size"]
        self.image_tensors = image_tensors
        self.dummy = dummy

    async def load_model(self) -> None:
        for part in ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]:
            db = await get_db()
            item_part = await find_by_item_idxs(part, db)
            self.item_parts.append(item_part)

        self.model = AutoEncoderPredictor(
            config=self.model_config,
            dropout_prop=self.model_config["dropout_prop"],
        )

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        print(self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    async def diagnosis(self, equips: Dict[str, int]):
        images = []

        for equip_part, equip_index in equips.items():
            if equip_index == -1:
                equip_index = self.dummy[self.dummy["equip_category"] == equip_part][
                    "index"
                ].values[0]

            images.append(self.image_tensors[equip_index])

        images = torch.stack(images).unsqueeze(0).to(self.device)
        score = self.model(images)
        return float(score)

    @torch.no_grad()
    async def get_topk_codi(self, equips: Dict[str, int]) -> List[List[int]]:
        # 한벌옷은 top으로 취급.
        # "Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon" 순서
        dummy_to_idx = {
            0: 10093,
            1: 10094,
            2: 10095,
            3: 10097,
            4: 10098,
            5: 10099,
            6: 10100,
        }

        part_index_and_topk = dict()

        # 착용하지 않은 부위 모두 dummy로 변환
        equips_list = list(equips.values())
        equips_list = [
            dummy_to_idx[i] if equips_list[i] == -1 else equips_list[i]
            for i in range(7)
        ]

        # 부위 별 top k 탐색 시작
        for part_idx, equip in enumerate(equips_list):
            part_scores = list()

            # 더미데이터 인 경우만 봐야함
            if equip not in dummy_to_idx.values():  # Fix item
                continue

            # 해당 부위의 모든 아이템
            item_part = self.item_parts[part_idx]

            # 저 중에 특정 비율만 보자. 나는 그냥 임의로 50%만 봤다.
            dataset_per_part = AETopkDataset(equips_list, part_idx, item_part)
            dataloader_per_part = DataLoader(
                dataset=dataset_per_part, batch_size=self.batch_size, shuffle=False,
            )

            for batch_idx, batch in enumerate(dataloader_per_part):
                # batch shape (batch, 7) == (16, 7)
                images = self.image_tensors[batch]
                output = self.model(images.transpose(0, 1).to(self.device))
                result = [
                    (batch[i, part_idx], float(output[i]))
                    for i in range(batch.shape[0])
                ]
                part_scores.extend(result)

            part_scores.sort(key=lambda x: x[1], reverse=True)
            part_index_and_topk[part_idx] = [x[0] for x in part_scores[:self.top_k]]

        # topk 개 다 뽑았으니, 경우의 수 고려 시작
        dataset_for_product = AEProductDataset(equips_list, part_index_and_topk)
        dataloader_for_product = DataLoader(
            dataset=dataset_for_product, batch_size=self.batch_size, shuffle=False,
        )

        codi_scores = []
        for batch_idx, batch in enumerate(dataloader_for_product):
            # batch shape (batch, 7) == (16, 7)
            images = self.image_tensors[batch]
            output = self.model(images.transpose(0, 1).to(self.device))
            result = [(batch[i, :], float(output[i])) for i in range(batch.shape[0])]
            codi_scores.extend(result)

        codi_scores.sort(key=lambda x: x[1], reverse=True)
        random_index = [0] + random.sample((range(1, len(codi_scores) - 1)), 2)

        return [codi_scores[x][0].detach().cpu().numpy().tolist() for x in random_index]
