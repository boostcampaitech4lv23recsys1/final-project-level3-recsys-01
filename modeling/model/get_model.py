import torch.nn as nn
from typing import Dict, Any

import modeling.model as models


def get_models(config: Dict[str, Any]) -> nn.Module:
    if config["arch"]["type"] == "NewMF":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            model_config["n_items"], model_config["n_factors"]
        )
    else:
        raise NotImplementedError
    return model
