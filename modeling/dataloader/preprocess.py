import pandas as pd

from typing import Dict, Any

from utils import GCSHelper, BigQueryHelper


class Preprocess:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.train_data = None
        self.gcs_helper = GCSHelper(
            key_path="keys/gcs_key.json",
            bucket_name="maple_preprocessed_data",
        )
        self.big_query_helper = BigQueryHelper(
            key_path="keys/gcs_key.json", dataset_name="train_dataset"
        )

    def load_data(self, is_train: bool = False) -> pd.DataFrame:
        print("---------------------------LOAD DATA FROM GCP-------------------------")
        if is_train:
            return self.big_query_helper.read_df_from_table(table_name="newMF")

        return self.gcs_helper.read_df_from_gcs(blob_name="item_KMST_1149_latest.csv")
