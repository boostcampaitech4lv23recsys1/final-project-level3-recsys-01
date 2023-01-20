from utilities import saving_text_file
from varname import nameof
import pandas as pd
import sys
import os
import re

# to import ../../utils.py
sys.path.append(
    os.path.dirname(
        os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    )
)
from utils import GCSHelper


class Preprocess:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.cfg_preprocess = config["preprocess"]
        self.save_dir = self.cfg_preprocess["idx_save_dir"]
        self.user_feature_engineering = self.cfg_preprocess["user_feature_engineering"]
        self.item_category = self.cfg_preprocess["item_category"]
        self.item_subCategory = self.cfg_preprocess["item_subCategory"]
        self.train_data = None
        self.gcs_helper = GCSHelper(
            "/opt/ml/final-project-level3-recsys-01/keys/gcs_key.json",
            "maple_preprocessed_data",
        )

    def __feature_engineering(
        self, data: pd.DataFrame, is_items: bool = False
    ) -> pd.DataFrame:
        if is_items:
            print("...item feature engineering...")
            # you can add items feature engineering if you want
        else:
            print("...user feature engineering...")
            data = data[self.user_feature_engineering]
        return data

    def __preprocessing(
        self, data: pd.DataFrame, items: pd.DataFrame, is_train: bool = False
    ) -> pd.DataFrame:
        user_detail = data
        print("--------------------------DATA PREPROCESSING--------------------------")
        print(f"...all item indexing...")
        item2idx = {k: i for i, k in enumerate(items["name_processed"].unique())}
        item2idx["-"] = len(item2idx)
        idx2item = {i: k for i, k in enumerate(items["name_processed"].unique())}
        idx2item[len(idx2item)] = "-"
        self.config["arch"]["args"]["n_items"] = len(item2idx)
        if is_train:
            saving_text_file(self.save_dir, item2idx, nameof(item2idx))
            saving_text_file(self.save_dir, idx2item, nameof(idx2item))
        print(f"Total items: {len(items)}")

        print("...user's item indexing...")
        user_detail = (
            user_detail.stack()
            .reset_index(level=1, drop=True)
            .to_frame()
            .rename(columns={0: "item"})
        )

        user_detail["item"] = user_detail["item"].apply(
            lambda x: x.replace("-", "empty") if x == "-" else x
        )
        pattern = r"[!@#$%^&*()_+=`~,.<>/?{}\s\[\]0123456789\-:]"
        user_detail["item"] = user_detail["item"].apply(
            lambda x: re.sub(pattern, "", x.lower())
        )
        print(f"Total users: {len(data)}")

        print("...Excluding outlier users...")
        not_in_maplestory_io = set()
        for user_detail_item_unique in user_detail["item"].unique():
            if user_detail_item_unique not in item2idx:
                not_in_maplestory_io.add(user_detail_item_unique)
        not_in_maplestory_io.remove("empty")
        user_detail["item"] = user_detail["item"].apply(
            lambda x: x.replace("empty", "-") if x == "empty" else x
        )
        user_detail["item"] = user_detail["item"].apply(
            lambda x: item2idx[x] if x in item2idx else x
        )
        drop_idx = user_detail[
            user_detail["item"].isin(not_in_maplestory_io)
        ].index.to_list()
        print(f"...user's item indexing...")
        data = (
            user_detail.reset_index()
            .groupby("index")
            .apply(lambda x: x["item"].to_list())
        )
        data = data.drop(drop_idx)
        print(f"total users: {len(data)} ")

        self.config["arch"]["args"]["n_users"] = len(data)
        if is_train:
            user_item_len = [
                self.config["arch"]["args"]["n_users"],
                self.config["arch"]["args"]["n_items"],
            ]
            saving_text_file(
                self.save_dir,
                user_item_len,
                nameof(user_item_len),
            )
        return data

    def load_data_from_file(self, is_items: bool = False):
        if is_items:
            print("...load item data from gcs...")
            df = self.gcs_helper.read_df_from_gcs(
                "item_KMST_1149_VER1.2.csv"
            )  # gcs_item_csv
        else:
            print("...load user data from gcs...")
            df = self.gcs_helper.read_df_from_gcs(
                "user_detail_VER1.0.csv"
            )  # gcs_user_csv
        return df

    def load_data(self, is_train: bool = False) -> pd.DataFrame:
        print("---------------------------LOAD DATA FROM GCS-------------------------")
        self.items_data = self.load_data_from_file(is_items=True)
        if is_train:
            self.items_data = self.__feature_engineering(self.items_data, is_items=True)

            self.users_data = self.load_data_from_file()
            self.users_data = self.__feature_engineering(self.users_data)
            self.users_data = self.__preprocessing(
                self.users_data, self.items_data, is_train=True
            )
            print(
                "number of users: ",
                self.config["arch"]["args"]["n_users"],
                "  number of items: ",
                self.config["arch"]["args"]["n_items"] - 1,  # except "-"
            )
            return self.users_data  # train
        return self.items_data  # inference
