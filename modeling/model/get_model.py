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
