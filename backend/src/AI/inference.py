from src.AI.models.model_object import NewMF, MCN
from src.utils.gcs_helper import GCSHelper
from src.database.models.crud_item import find_by_item_idxs
from src.database.init_db import get_db
from src.AI.config import MODEL_CONFIG
from src.AI.image_processing import image_to_tensor

import os
import torch
import asyncio


class InferenceNewMF(object):
    def __init__(self, model_config):
        self.model_config = model_config
        self.model_path = model_config["model_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(self.model_path):
            gcs_helper = GCSHelper(
                key_path="src/utils/gcs_key.json", bucket_name="maple_trained_model"
            )
            gcs_helper.download_file_from_gcs(
                blob_name="NewMF/NewMF_latest.pt", file_name=self.model_path
            )
        self.item_parts = list()
        self.n_items = 0

    @torch.no_grad()
    async def inference(self, equips):
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

    async def load_model(self):
        for part in ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]:
            db = await get_db().__anext__()
            item_part = await find_by_item_idxs(part, db)
            self.n_items += len(item_part)
            self.item_parts.append(item_part)

        self.model = NewMF(
            n_items=self.n_items, n_factors=self.model_config["n_factors"]
        )
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
    def __init__(self, model_config):
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

        self.model = MCN(
            embed_size=model_config["embed_size"],
            need_rep=model_config["need_rep"],
            vocabulary=None,
            vse_off=model_config["vse_off"],
            pe_off=model_config["pe_off"],
            mlp_layers=model_config["mlp_layers"],
            conv_feats=model_config["conv_feats"],
            pretrained=model_config["pretrained"],
        )
        self.top_k = model_config["top_k"]

    async def load_model(self):
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        print(self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    async def diagnosis(self, equips: dict):
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


async def get_model_newMF():
    try:
        yield MODELS["newMF"]
    finally:
        pass
