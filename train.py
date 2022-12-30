from dataloader.preprocess import Preprocess
import argparse
import torch
import model as models
from trainer import newMFTrainer
from utils import read_json, set_seed

def main(config):
    preprocess = Preprocess(config)
    data = preprocess.load_train_data()
    # buwi = preprocess.item_buwi_list(data)

    model = models.get_models(config)


    trainer = newMFTrainer(model, data, config)

    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="DKT Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="./mfconfig.json",
        type=str,
        help='config 파일 경로 (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["trainer"]["seed"])

    main(config)

    


    