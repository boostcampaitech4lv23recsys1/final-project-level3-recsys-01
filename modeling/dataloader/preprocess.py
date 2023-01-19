from utilities import saving_text_file

from varname import nameof
import pandas as pd
import json
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
            key_path = "/opt/ml/final-project-level3-recsys-01/keys/gcs_key.json",
            bucket_name = "maple_preprocessed_data"
        )

    def __feature_engineering(self, data: pd.DataFrame, is_items=False) -> pd.DataFrame:
        if is_items:
            print(
                "------------------------item feature engineering----------------------"
            )
            data = data  # 이미 전처리 된 아이템 data가 입력
            # data = data.drop_duplicates(subset="name")
            # data = data[
            #     (data["category"].isin(self.item_category))
            #     & (data["subCategory"].isin(self.item_subCategory))
            # ]
            # data = data.drop_duplicates(subset=["name"])  # item drop duplicates
            # self.config["arch"]["args"]["n_items"] = len(data)
        else:
            print(
                "------------------------user feature engineering----------------------"
            )
            data = data[self.user_feature_engineering]
        print("using columns: ", list(data.columns))
        return data

    def __preprocessing(
        self, data: pd.DataFrame, items: pd.DataFrame, is_train=False
    ) -> pd.DataFrame:
        user_detail = data
        print("--------------------------data preprocessing--------------------------")
        print(f"...item indexing... all unique items: {len(items)}")
        item2idx = {k: i for i, k in enumerate(items["name_processed"].unique())}
        item2idx["-"] = len(item2idx)
        idx2item = {i: k for i, k in enumerate(items["name_processed"].unique())}
        idx2item[len(idx2item)] = "-"
        self.config["arch"]["args"]["n_items"] = len(item2idx)

        if is_train:
            saving_text_file(self.save_dir, item2idx, nameof(item2idx))
            saving_text_file(self.save_dir, idx2item, nameof(idx2item))

        print("...users item indexing... [1/2] disassemble")
        print(f"total users: {len(data)}")
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
        print("...except certain users...")
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
        print(f"...users item indexing... [2/2] assemble")
        # print("...listing items by user...")
        # user_detail = user_detail.stack().reset_index(level=1, drop=True).to_frame().rename(columns={0: "item"})

        # pattern = r"[!@#$%^&*()_+=`~,.<>/?{}\s\[\]0123456789-]"
        # users_item_list = [data[ufe] for ufe in self.user_feature_engineering]
        # user_test = pd.concat(users_item_list).reset_index(drop=True).apply(lambda x: re.sub(pattern, "", x.lower()))
        # user_test = user_test[user_test != ""].unique()

        data = (
            user_detail.reset_index()
            .groupby("index")
            .apply(lambda x: x["item"].to_list())
        )
        data = data.drop(drop_idx)
        print(f"total users: {len(data)} ")

        self.config["arch"]["args"]["n_users"] = len(data)  # all users
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

    # def __preprocessing(
    #     self, data: pd.DataFrame, items: pd.DataFrame, is_train=False
    # ) -> pd.DataFrame:
    #     print("--------------------------data preprocessing--------------------------")
    #     print("...listing items by user...")
    #     user_detail = data
    #     user_detail = (
    #         user_detail.stack()
    #         .reset_index(level=1, drop=True)
    #         .to_frame()
    #         .rename(columns={0: "item"})
    #     )
    #     print("...item indexing...")
    #     item2idx = {k: i for i, k in enumerate(user_detail["item"].unique())}
    #     idx2item = {i: k for i, k in enumerate(user_detail["item"].unique())}
    #     if is_train:
    #         saving_text_file(self.save_dir, item2idx, nameof(item2idx))
    #         saving_text_file(self.save_dir, idx2item, nameof(idx2item))

    #         # item2idx = items[["name", "id"]].set_index("name").to_dict()["id"]
    #         print("...item to idx by user...")
    #         user_detail["item"] = user_detail["item"].apply(lambda x: item2idx[x])
    #         data = (
    #             user_detail.reset_index()
    #             .groupby("index")
    #             .apply(lambda x: x["item"].to_list())
    #         )

    #     self.config["arch"]["args"]["n_users"] = len(user_detail)  # all users

    #     return data

    # def item_buwi_list(self, data: pd.DataFrame):
    #     user_detail = data
    #     print(user_detail)

    #     hat = user_detail[user_detail['codi-hat'] != "-"]['codi-hat'].unique()
    #     hair = user_detail[user_detail['codi-hair'] != "-"]['codi-hair'].unique()
    #     face = user_detail[user_detail['codi-face'] != "-"]['codi-face'].unique()
    #     top = user_detail[user_detail['codi-top'] != "-"]['codi-top'].unique()
    #     bottom = user_detail[user_detail['codi-bottom'] != "-"]['codi-bottom'].unique()
    #     shoes = user_detail[user_detail['codi-shoes'] != "-"]['codi-shoes'].unique()
    #     weapon = user_detail[user_detail['codi-weapon'] != "-"]['codi-weapon'].unique()

    #     return [hat, hair, face, top, bottom, shoes, weapon]

    def load_data_from_file(self, is_items=False):
        if is_items:
            print(
                "------------------------load item data from gcs-----------------------"
            )
            df = self.gcs_helper.read_df_from_gcs(
                "item_KMST_1149_VER1.2.csv"
            )  # gcs_item_csv
        else:
            print(
                "------------------------load user data from gcs-----------------------"
            )
            df = self.gcs_helper.read_df_from_gcs(
                "user_detail_VER1.0.csv"
            )  # gcs_user_csv
        return df

    def load_train_data(self) -> pd.DataFrame:
        self.items_data = self.load_data_from_file(is_items=True)
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
            self.config["arch"]["args"]["n_items"],
        )
        return self.users_data

    def load_test_data(self) -> pd.DataFrame:
        self.items_data = self.load_data_from_file(is_items=True)
        # self.items_data = self.__feature_engineering(self.items_data, is_items=True)

        # self.users_data = self.load_data_from_file()
        # self.users_data = self.__feature_engineering(self.users_data)
        # self.users_data = self.__preprocessing(self.users_data, self.items_data)
        # print(
        #     "number of users: ",
        #     self.config["arch"]["args"]["n_users"],
        #     "  number of items: ",
        #     self.config["arch"]["args"]["n_items"],
        # )
        return self.items_data
