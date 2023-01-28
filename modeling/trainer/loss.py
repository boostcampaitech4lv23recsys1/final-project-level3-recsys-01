import torch.nn.functional as F
import torch.nn as nn

def BCE_loss(output, target):
    loss = nn.BCELoss()
    return loss(output, target)

def get_loss(trainer_config):
    if trainer_config["loss"] == "bce":
        return BCE_loss