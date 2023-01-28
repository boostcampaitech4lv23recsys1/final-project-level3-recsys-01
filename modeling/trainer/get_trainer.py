import torch.nn as nn
from typing import Dict, Any

import modeling.trainer as trainers


def get_trainers(config: Dict[str, Any], model, train_dataloader, val_dataloader):
    if config["arch"]["type"] == "NewMF":
        trainer = getattr(trainers, "newMFTrainer")(
            config=config,
            model=model,
            train_data_loader=train_dataloader,
            valid_data_loader=val_dataloader
        )
    elif config["arch"]["type"] == "MCN":
        trainer = getattr(trainers, "MCNTrainer")(
            config=config,
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader
        )
    else:
        raise NotImplementedError
    return trainer
