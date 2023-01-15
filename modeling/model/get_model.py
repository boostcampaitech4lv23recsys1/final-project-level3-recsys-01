import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import model as models


def get_models(config):
    if config["arch"]["type"] == "NewMF":
        model_config = config["arch"]["args"]
        model = getattr(models, config["arch"]["type"])(
            model_config["n_users"], model_config["n_items"], model_config["n_factors"]
        )
    else:
        raise NotImplementedError
    return model