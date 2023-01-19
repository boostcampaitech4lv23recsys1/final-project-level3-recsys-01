from sklearn.model_selection import train_test_split
from collections import OrderedDict
from ast import literal_eval
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
import json
import os


def data_split(config: OrderedDict, data: pd.DataFrame) -> pd.DataFrame:
    test_size = config["dataset"]["test_size"]
    shuffle = config["dataset"]["shuffle"]
    X_train, X_valid = train_test_split(data, test_size=test_size, shuffle=shuffle)
    return X_train, X_valid


def read_json(fname: str) -> OrderedDict:
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def set_seed(seed: int = 417) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def saving_text_file(dir: str, file: object, filename: str) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = open(f"{dir}/{filename}.txt", "w")
    f.write(str(file))
    f.close


def loading_text_file(filename: str) -> object:
    f = open(f"./temporary/{filename}.txt", "r")
    strings = f.read()
    return literal_eval(strings)


# def loading_text_file(filename):
#     f = open(f"./temporary/{filename}.txt", "r")
#     strings = f.read()[1:-1]
#     strings = strings.replace("'", "")
#     strings_list = strings.split(", ")
#     key, value = [], []
#     if filename == "idx2item":
#         for string in strings_list:
#             pair = string.split(": ")
#             key.append(int(pair[0]))
#             value.append(pair[1])
#     elif filename == "item2idx":
#         for string in strings_list:
#             pair = string.split(": ")
#             key.append(pair[0])
#             value.append(int(pair[1]))
#     return dict(zip(key, value))
