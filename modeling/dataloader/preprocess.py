import pandas as pd
import os

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
            return self.big_query_helper.read_df_from_table(
                table_name=f"{self.config['arch']['type']}"
            )

        return self.gcs_helper.read_df_from_gcs(blob_name="item_KMST_1149_latest.csv")

    def download_images(self) -> None:
        if os.path.exists("modeling/data/image/item"):
            print("----------데이터가 이미 저장되어 있습니다.----------")
            return

        print("----------이미지 데이터 다운로드를 시작합니다. ----------")
        self.gcs_helper.change_bucket("maple_raw_data")
        self.gcs_helper.download_folder_from_gcs(
            folder_name="image/item", save_path="modeling/data/image/item"
        )
