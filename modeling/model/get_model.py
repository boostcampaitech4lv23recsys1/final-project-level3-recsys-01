import model as models
import torch.nn as nn


def get_models(config: dict) -> nn.Module:
    if config["arch"]["type"] == "NewMF":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            model_config["n_users"], model_config["n_items"], model_config["n_factors"]
        )
    else:
        raise NotImplementedError
    return model
