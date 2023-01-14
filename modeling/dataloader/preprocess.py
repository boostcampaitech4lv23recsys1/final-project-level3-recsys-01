from utilities import saving_text_file

from varname import nameof
import pandas as pd
import sys
import os

# to import ../../utils.py
sys.path.append(
    os.path.dirname(
        os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    )
)
from utils import GCS_helper


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_preprocess = config["preprocess"]
        self.save_dir = self.cfg_preprocess["idx_save_dir"]
        self.user_feature_engineering = self.cfg_preprocess["user_feature_engineering"]
        self.item_feature_engineering = self.cfg_preprocess["item_feature_engineering"]
        self.item_feature_engineering_weapon = self.cfg_preprocess[
            "item_feature_engineering_weapon"
        ]
        self.train_data = None
        self.gcs_helper = GCS_helper(
            "/opt/ml/final-project-level3-recsys-01/keys/gcs_key.json"
        )

    def __feature_engineering(self, data: pd.DataFrame, is_items=False):
        if is_items:
            print(
                "------------------------item feature engineering----------------------"
            )
            data = data[
                (data["category"].isin(self.item_feature_engineering))
                & (data["subCategory"].isin(self.item_feature_engineering))
                | (data["subCategory"].isin(self.item_feature_engineering_weapon))
            ]
            data = data.drop_duplicates(subset=["name"])  # item drop duplicates
            self.config["arch"]["args"]["n_items"] = len(data)
        else:
            print(
                "------------------------user feature engineering----------------------"
            )
            data = data[self.user_feature_engineering]
        print("using columns: ", list(data.columns))
        return data

    def __preprocessing(self, data: pd.DataFrame, items: pd.DataFrame, is_train=False):
        print("--------------------------data preprocessing--------------------------")
        print("...listing items by user...")
        user_detail = data
        user_detail = (
            user_detail.stack()
            .reset_index(level=1, drop=True)
            .to_frame()
            .rename(columns={0: "item"})
        )
        print("...item indexing...")
        item2idx = {k: i for i, k in enumerate(user_detail["item"].unique())}
        idx2item = {i: k for i, k in enumerate(user_detail["item"].unique())}
        if is_train:
            saving_text_file(self.save_dir, item2idx, nameof(item2idx))
            saving_text_file(self.save_dir, idx2item, nameof(idx2item))

            # item2idx = items[["name", "id"]].set_index("name").to_dict()["id"]
            print("...item to idx by user...")
            user_detail["item"] = user_detail["item"].apply(lambda x: item2idx[x])
            data = (
                user_detail.reset_index()
                .groupby("index")
                .apply(lambda x: x["item"].to_list())
            )

        self.config["arch"]["args"]["n_users"] = len(user_detail)  # all users

        return data

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
                "csv/new_maple_item.csv"
            )  # gcs_item_csv
        else:
            print(
                "------------------------load user data from gcs-----------------------"
            )
            df = self.gcs_helper.read_df_from_gcs(
                "csv/user_detail_total.csv"
            )  # gcs_user_csv
        return df

    def load_train_data(self):
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

    def load_test_data(self):
        self.items_data = self.load_data_from_file(is_items=True)
        self.items_data = self.__feature_engineering(self.items_data, is_items=True)

        self.users_data = self.load_data_from_file()
        self.users_data = self.__feature_engineering(self.users_data)
        self.users_data = self.__preprocessing(self.users_data, self.items_data)
        print(
            "number of users: ",
            self.config["arch"]["args"]["n_users"],
            "  number of items: ",
            self.config["arch"]["args"]["n_items"],
        )
        return self.items_data
