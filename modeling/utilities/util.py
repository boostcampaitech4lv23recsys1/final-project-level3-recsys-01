from pathlib import Path
import json
import os
import random
import numpy as np
import torch
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
