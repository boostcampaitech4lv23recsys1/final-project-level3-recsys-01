import torch

import argparse
from typing import Dict, Any

from modeling.dataloader import Preprocess, get_loader
import modeling.model as models
from modeling.dataset import get_datasets
from modeling.trainer import get_trainers
from modeling.utilities import read_json, set_seed, data_split

import wandb

from pytz import timezone
from datetime import datetime


def main(config: Dict[str, Any]) -> None:
    preprocess = Preprocess(config)
    inter_data = preprocess.load_data(is_train=True)
    item_data = preprocess.load_data(is_train=False)

    print(f"len inter_data: {inter_data.shape[0]}...")
    print(f"len item_data: {item_data.shape[0]}...")
    if config["arch"]["type"] in ["MCN", "SimpleMCN"]:
        preprocess.download_images()

    config["arch"]["args"]["n_items"] = item_data.shape[0]

    train_data, valid_data = data_split(config, inter_data)

    train_set = get_datasets(config, train_data, item_data)
    valid_set = get_datasets(config, valid_data, item_data, is_train=False)

    train_loader, valid_loader = get_loader(config, train_set, valid_set)

    model = models.get_models(config)
    trainer = get_trainers(config, model, train_loader, valid_loader)

    now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y-%m-%d_%H:%M")
    wandb.init(
        project=config["arch"]["type"],
        entity="dino-final",
        name=f"{now}_{args.user}",
    )
    wandb.define_metric("train/*", step_metric="batch_num")
    wandb.define_metric("train_epoch/*", step_metric="train_step")
    wandb.define_metric("valid/*", step_metric="valid_step")
    wandb.watch(model)

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
    args.add_argument("--user", type=str, default=None)
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["GCS_upload"] = args.gcs
    set_seed(config["seed"])

    main(config)
