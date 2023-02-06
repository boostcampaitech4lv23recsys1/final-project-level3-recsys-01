from src.AI.models.model_object import NewMF, MCN, MCNTopkDataset, MCNProductDataset
from src.utils.gcs_helper import GCSHelper
from src.database.models.crud_item import find_by_item_idxs
from src.database.init_db import get_db
from src.AI.config import MODEL_CONFIG
from src.AI.image_processing import image_to_tensor

import torch
from torch.utils.data import DataLoader

import os
import asyncio
import random

from typing import Dict, List, Any


class InferenceNewMF(object):
    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.model_config = model_config
        self.model_path = model_config["model_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(self.model_path):
            os.makedirs("/".join(self.model_path.split("/")[:-1]))
            gcs_helper = GCSHelper(
                key_path="src/utils/gcs_key.json", bucket_name="maple_trained_model"
            )
            gcs_helper.download_file_from_gcs(
                blob_name="NewMF/NewMF_latest.pt", file_name=self.model_path
            )
        self.item_parts = list()
        self.n_items = 0

    @torch.no_grad()
    async def inference(self, equips: Dict[str, int]):
        predicts = list()
        equips = list(equips.values())
        for part_idx, equip in enumerate(equips):
            if equip != -1:
                predicts.append([(equip, 1) for _ in range(self.top_k)])
            else:
                part_scores = list()

                item_part = self.item_parts[part_idx]
                for any_item_in_part in item_part:
                    temp_equips = equips[:]

                    if part_idx < 4:
                        temp_equips[part_idx] = any_item_in_part
                    else:
                        temp_equips[part_idx - 1] = any_item_in_part

                    output = self.model(torch.tensor([temp_equips]).to(self.device))

                    part_scores.append((any_item_in_part, float(output)))
                part_scores.sort(key=lambda x: x[1], reverse=True)
                part_recommendation = part_scores[: self.top_k]
                predicts.append(part_recommendation)

        return predicts

    async def load_model(self) -> None:
        for part in ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]:
            db = await get_db().__anext__()
            item_part = await find_by_item_idxs(part, db)
            self.n_items += len(item_part)
            self.item_parts.append(item_part)

        self.model = NewMF(n_items=10101, n_factors=self.model_config["n_factors"])
        load_state = torch.load(self.model_path, map_location=self.device)
        print(self.device)
        self.model.load_state_dict(load_state["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.top_k = self.model_config["top_k"]


is_load = True
if is_load:
    image_tensors, item_data = asyncio.run(image_to_tensor())
    dummy = item_data[item_data["category"] == "dummy"]


class MCNInference:
    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.model_config = model_config
        self.model_path = model_config["model_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(self.model_path):
            gcs_helper = GCSHelper(
                key_path="src/utils/gcs_key.json", bucket_name="maple_trained_model"
            )
            gcs_helper.download_file_from_gcs(
                blob_name="MCN/MCN_latest.pt", file_name=self.model_path
            )

        self.item_parts = list()
        self.top_k = model_config["top_k"]
        self.batch_size = model_config["batch_size"]

    async def load_model(self) -> None:
        for part in ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]:
            db = await get_db().__anext__()
            item_part = await find_by_item_idxs(part, db)
            self.item_parts.append(item_part)

        self.model = MCN(
            embed_size=self.model_config["embed_size"],
            need_rep=self.model_config["need_rep"],
            vocabulary=None,
            vse_off=self.model_config["vse_off"],
            pe_off=self.model_config["pe_off"],
            mlp_layers=self.model_config["mlp_layers"],
            conv_feats=self.model_config["conv_feats"],
            pretrained=self.model_config["pretrained"],
            resnet_layer_num=self.model_config["resnet_layer_num"],
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
                equip_index = dummy[dummy["equip_category"] == equip_part][
                    "index"
                ].values[0]

            images.append(image_tensors[equip_index])

        images = torch.stack(images).unsqueeze(0).to(self.device)
        score, _, __, ___ = self.model.forward(images)
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
            dataset_per_part = MCNTopkDataset(equips_list, part_idx, item_part)
            dataloader_per_part = DataLoader(
                dataset=dataset_per_part, batch_size=self.batch_size, shuffle=False
            )

            for batch_idx, batch in enumerate(dataloader_per_part):
                # batch shape (batch, 7) == (16, 7)
                images = image_tensors[batch]
                output, _, _, _ = self.model(images.to(self.device))
                result = [
                    (batch[i, part_idx], float(output[i]))
                    for i in range(batch.shape[0])
                ]
                part_scores.extend(result)

            part_scores.sort(key=lambda x: x[1], reverse=True)
            part_index_and_topk[part_idx] = [x[0] for x in part_scores[: self.top_k]]

        # topk 개 다 뽑았으니, 경우의 수 고려 시작
        dataset_for_product = MCNProductDataset(equips_list, part_index_and_topk)
        dataloader_for_product = DataLoader(
            dataset=dataset_for_product, batch_size=self.batch_size, shuffle=False
        )

        codi_scores = []
        for batch_idx, batch in enumerate(dataloader_for_product):
            # batch shape (batch, 7) == (16, 7)
            images = image_tensors[batch]
            output, _, _, _ = self.model(images.to(self.device))
            result = [(batch[i, :], float(output[i])) for i in range(batch.shape[0])]
            codi_scores.extend(result)

        codi_scores.sort(key=lambda x: x[1], reverse=True)
        random_index = [0] + random.sample((range(1, len(codi_scores) - 1)), 2)

        return [codi_scores[x][0].tolist() for x in random_index]


if is_load:
    newMF = InferenceNewMF(model_config=MODEL_CONFIG["newMF"])
    asyncio.run(newMF.load_model())
    mcn = MCNInference(model_config=MODEL_CONFIG["MCN"])
    asyncio.run(mcn.load_model())
    MODELS = {"newMF": newMF, "MCN": mcn}
    is_load = False


async def get_model():
    try:
        yield MODELS["MCN"]
    finally:
        pass
