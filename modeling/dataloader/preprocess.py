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

        self.feature_engineering = [
            "codi-hat",
            "codi-hair",
            "codi-face",
            "codi-top",
            "codi-bottom",
            "codi-shoes",
            "codi-weapon",
        ]

        self.train_data = None
        self.gcs_helper = GCS_helper(
            "/opt/ml/final-project-level3-recsys-01/keys/gcs_key.json"
        )

    def __feature_engineering(self, data: pd.DataFrame):
        print("--------------------------feature engineering-------------------------")
        data = data[self.feature_engineering]
        return data

    def __preprocessing(self, data: pd.DataFrame, is_train=True):
        print("--------------------------data preprocessing--------------------------")
        user_detail = data
        user_detail
        user_detail = (
            user_detail.stack()
            .reset_index(level=1, drop=True)
            .to_frame()
            .rename(columns={0: "item"})
        )

        item2idx = {k: i for i, k in enumerate(user_detail["item"].unique())}

        user_detail["item"] = user_detail["item"].apply(lambda x: item2idx[x])
        data = (
            user_detail.reset_index()
            .groupby("index")
            .apply(lambda x: x["item"].to_list())
        )

        self.config["arch"]["args"]["n_users"] = len(user_detail)  # all users
        self.config["arch"]["args"]["n_items"] = len(
            item2idx
        )  # not all items, please fix it

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

    def load_data_from_file(self):
        print("--------------------------load data from gcs--------------------------")
        df = self.gcs_helper.read_df_from_gcs(
            "csv/user_detail_temp_total.csv"
        )  # gcs_csv
        # df = pd.read_csv( # 일단 한 csv 파일만 가져와보자
        #   f"{self.cfg_preprocess['data_dir']}/user_detail/user_detail_{str(self.cfg_preprocess['start'])}_{str(self.cfg_preprocess['start'] + 10000)}.csv"
        # )
        return df

    def load_train_data(self):
        self.train_data = self.load_data_from_file()
        self.train_data = self.__feature_engineering(self.train_data)
        self.train_data = self.__preprocessing(self.train_data, is_train=True)
        return self.train_data
