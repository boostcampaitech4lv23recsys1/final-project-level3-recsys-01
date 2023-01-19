from pathlib import Path
import json
import os
import random
import numpy as np
import torch
from ast import literal_eval
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def data_split(config, data):
    test_size = config["dataset"]["test_size"]
    shuffle = config["dataset"]["shuffle"]
    X_train, X_valid = train_test_split(data, test_size=test_size, shuffle=shuffle)
    return X_train, X_valid


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def set_seed(seed=417):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def saving_text_file(dir, file, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = open(f"{dir}/{filename}.txt", "w")
    f.write(str(file))
    f.close


def loading_text_file(filename):
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
