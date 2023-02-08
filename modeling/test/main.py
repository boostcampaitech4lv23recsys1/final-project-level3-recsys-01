import torch
from torch.utils.data import DataLoader

from typing import Dict, Any

import modeling.model as models
from modeling.test.load_data import load_test_data, download_images
from modeling.test.tester import Tester
from modeling.test.test_dataset import TestDataset


def main(config: Dict[str, Any]) -> None:
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    inter_data, item_data = load_test_data()

    print(f"len inter_data: {inter_data.shape[0]}...")
    print(f"len item_data: {item_data.shape[0]}...")

    image_tensors = None
    if config["arch"]["type"] in ["MCN", "SimpleMCN"]:
        image_tensors = download_images(item_data)

    config["arch"]["args"]["n_items"] = item_data.shape[0]

    test_set = TestDataset(inter_data, item_data)

    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = models.get_models(config)
    model_path = "modeling/save_models/SimpleMCN_20230207-1633.pt"
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))

    #
    tester = Tester(config, model, test_loader, item_data, image_tensors)

    tester.test()
