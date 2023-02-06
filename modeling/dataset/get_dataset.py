from torch.utils.data import Dataset
from pandas import DataFrame
from typing import Dict, Any

import modeling.dataset as datasets


def get_datasets(
    config: Dict[str, Any], inter_data: DataFrame, item_data: DataFrame
) -> Dataset:
    available_models = ["NewMF", "MCN"]
    model_name = config["arch"]["type"]

    if model_name not in available_models:
        raise NotImplementedError(
            f"그런 모델은 없어요~ 입력한 모델: {model_name}, 가능한 모델: {available_models}"
        )

    return getattr(datasets, config["arch"]["type"])(
        inter_data=inter_data,
        item_data=item_data,
        negative_ratio=config["dataset"]["negative_ratio"],
        n_change_parts=config["dataset"]["n_change_parts"],
    )
