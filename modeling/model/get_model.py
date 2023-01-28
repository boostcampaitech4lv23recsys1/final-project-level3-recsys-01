import torch.nn as nn
from typing import Dict, Any

import modeling.model as models


def get_models(config: Dict[str, Any]) -> nn.Module:
    if config["arch"]["type"] == "NewMF":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            model_config["n_items"], model_config["n_factors"]
        )
    elif config["arch"]["type"] == "MCN":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            embed_size=1000,
            need_rep=True,
            vocabulary=config["dataloader"]["args"]["vocabulary_len"],
            vse_off=model_config["vse_off"],
            pe_off=model_config["pe_off"],
            mlp_layers=model_config["mlp_layers"],
            conv_feats=model_config["conv_feats"],
        )
    else:
        raise NotImplementedError
    return model
