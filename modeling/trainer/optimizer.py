from torch.optim import Adam, AdamW, SGD


def get_optimizer(model, config):
    if config['type'] == "adam":
        optimizer = Adam(params=model.parameters(), lr=config['args']['lr'], weight_decay=config['args']['weight_decay'])
    if config['type'] == "adamW":
        optimizer = AdamW(params=model.parameters(), lr=config['args']['lr'], weight_decay=config['args']['weight_decay'])
    if config["type"] == "sgd":
        optimizer = SGD(params=model.parameters(), lr=config["args"]["lr"], momentum=config["args"]["momentum"])

    return optimizer