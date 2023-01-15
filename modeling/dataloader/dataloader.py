from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


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
        collate_fn=default_collate,
    )
    valid_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=default_collate,
    )

    return train_loader, valid_loader
