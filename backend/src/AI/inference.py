from src.AI.models.model_object import NewMF
from src.utils.gcs_helper import GCSHelper
from src.database.models.crud_item import find_by_item_idxs
from src.database.init_db import get_db
from src.AI.config import MODEL_CONFIG

import os
import torch


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
        load_state = torch.load(self.model_path)
        self.model.load_state_dict(load_state["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.top_k = self.model_config["top_k"]


class MCNInference(object):
    def __init__(self):
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
        load_state = torch.load(self.model_path)
        self.model.load_state_dict(load_state["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.top_k = model_config["top_k"]


newMF = InferenceNewMF(model_config=MODEL_CONFIG["newMF"])
MODELS = {"newMF": newMF}


async def get_model():
    if newMF.n_items == 0:
        await newMF.load_model()
    try:
        yield MODELS["newMF"]
    finally:
        pass
