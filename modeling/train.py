from dataloader import Preprocess
from dataloader import get_loader
import argparse
import torch
import model as models
from dataset import BaseDataset
from trainer import newMFTrainer
from utilities import read_json, set_seed, data_split


def main(config: dict) -> None:
    preprocess = Preprocess(config)
    data = preprocess.load_train_data()

    model = models.get_models(config)
    train_data, valid_data = data_split(config, data)

    train_set = BaseDataset(train_data)
    valid_set = BaseDataset(valid_data)

    train_loader, valid_loader = get_loader(config["dataloader"]["args"], train_set, valid_set)
    trainer = newMFTrainer(config, model, train_loader, valid_loader)

    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Final Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="./config/mfconfig.json",
        type=str,
        help='config 파일 경로 입력 (default: "./config/mfconfig.json")',
    )
    args.add_argument("-g", "--gcs", default = False, type=bool, help='GCS 업로드 여부 선택 (default: False)')
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["GCS_upload"] = args.gcs
    set_seed(config["seed"])

    main(config)
    


    