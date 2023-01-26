import torch

import argparse
from typing import Dict, Any

from modeling.dataloader import Preprocess, get_loader
import modeling.model as models
from modeling.dataset import get_datasets
from modeling.trainer import newMFTrainer
from modeling.utilities import read_json, set_seed, data_split


def main(config: Dict[str, Any]) -> None:
    preprocess = Preprocess(config)
    data = preprocess.load_data(is_train=True)
    item_data = preprocess.load_data(is_train=False)

    config["arch"]["args"]["n_items"] = item_data.shape[0]

    train_data, valid_data = data_split(config, data)

    train_set = get_datasets(config, train_data)
    valid_set = get_datasets(config, valid_data)

    train_loader, valid_loader = get_loader(
        config["dataloader"]["args"], train_set, valid_set
    )

    model = models.get_models(config)
    trainer = newMFTrainer(config, model, train_loader, valid_loader)

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Final Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="modeling/config/mfconfig.json",
        type=str,
        help='config 파일 경로 입력 (default: "modeling/config/mfconfig.json")',
    )
    args.add_argument(
        "-g", "--gcs", default=False, type=bool, help="GCS 업로드 여부 선택 (default: False)"
    )
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["GCS_upload"] = args.gcs
    set_seed(config["seed"])

    main(config)
