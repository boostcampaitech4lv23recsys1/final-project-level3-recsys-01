from torch.utils.data import Dataset, DataLoader, default_collate
import torch

from typing import Dict, Tuple, Any


def get_loader(
    config: Dict[str, Any],
    train_set: Dataset,
    val_set: Dataset,
) -> Tuple[DataLoader, DataLoader]:
    """
    get Data Loader
    """
    dataloader_config = config["dataloader"]["args"]
    if config["arch"]["type"] == "MCN":
        collate_fn = mcn_collate_fn
    else:
        collate_fn = default_collate

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=dataloader_config["batch_size"],
        num_workers=dataloader_config["num_workers"],
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=dataloader_config["batch_size"],
        num_workers=dataloader_config["num_workers"],
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader


def mcn_collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images, is_compat = zip(*data)
    is_compat = torch.LongTensor(is_compat)
    images = torch.stack(images)
    return (
        images,
        is_compat
    )
