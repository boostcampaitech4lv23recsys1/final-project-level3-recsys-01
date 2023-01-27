from torch.utils.data import Dataset, DataLoader, default_collate

from typing import Dict, Tuple, Any


def get_loader(
    config: Dict[str, Any],
    train_set: Dataset,
    val_set: Dataset,
) -> Tuple[DataLoader, DataLoader]:
    """
    get Data Loader
    """
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=default_collate,
    )
    valid_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=default_collate,
    )

    return train_loader, valid_loader
