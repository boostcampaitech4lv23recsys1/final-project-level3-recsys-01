from torch.utils.data import Dataset, DataLoader, default_collate
import torch

def get_loader(
        config: dict,
        train_set: Dataset,
        val_set: Dataset,
) -> DataLoader:
    """
    get Data Loader
    """
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=default_collate
    )
    valid_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=default_collate
    )

    return train_loader, valid_loader


# def collate_fn(samples):
#     x = []
#     y = []
    
#     for sample in samples:
#         x.append(torch.tensor(sample["x"]))
#         y.append(torch.tensor(sample["y"]))
#     return {
#         "x": torch.stack(x),
#         "y": torch.stack(y)
#     }