from .main import main

import argparse
from modeling.utilities import read_json, set_seed


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Final Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="modeling/config/mfconfig.json",
        type=str,
        help='config 파일 경로 입력 (default: "modeling/config/mfconfig.json")',
    )

    args = args.parse_args()
    config = read_json(args.config)

    set_seed(config["seed"])

    main(config)
