from torch.optim.lr_scheduler import StepLR

def get_scheduler(optimizer, trainer_config):
    if "lr_scheduler" in trainer_config:
        if trainer_config["lr_scheduler"]["type"] == "steplr":
            scheduler_config = trainer_config["lr_scheduler"]
            return StepLR(
                optimizer,
                step_size=scheduler_config["args"]["step_size"],
                gamma=scheduler_config["args"]["gamma"]
            )