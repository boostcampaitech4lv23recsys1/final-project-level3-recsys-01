from utils import GCSHelper, BigQueryHelper
from preprocessing.user_MCN import (
    drop_columns,
    drop_exact_same,
    drop_na,
    fill_hat_and_shoes_as_transparency,
    item_name_processing,
    item_name_to_index,
)

import os
import pandas as pd
from datetime import datetime
from typing import Sequence, Tuple
import tarfile
import torchvision
from tqdm import tqdm
from PIL import Image
import torch

key_path = "keys/gcs_key.json"

gcs_helper = GCSHelper(key_path=key_path, bucket_name="maple_raw_data")

# 기준 시각 2023년 2월 4일 20시 55분 49초
def read_df_after_crawling(
    user_csv_path: str, last_update: Sequence[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """마지막으로 빅쿼리에 올린 시각 이후의 데이터들로 테스트 시작

    Args:
        user_csv_path (str): csv이름. ex) csv/user_detail_jeong.csv
        last_update (Sequence[int]): 년,월,일,시,분,초로 이루어진 sequence. ex) (2023, 2, 4, 20, 55, 49)

    Returns:
        _type_: DataFrame
    """

    gcs_helper.change_bucket("maple_raw_data")
    user_df = gcs_helper.read_df_from_gcs(user_csv_path)
    gcs_helper.change_bucket("maple_preprocessed_data")
    item_df = gcs_helper.read_df_from_gcs("item_KMST_1149_latest.csv")

    # 기준 시각 이후의 데이터만 선택하기
    user_df["updated_at"] = pd.to_datetime(user_df["updated_at"])
    last_update_datetime = datetime(*last_update)
    user_df = user_df[user_df["updated_at"] > last_update_datetime]

    user_df = drop_columns(user_df)
    user_df = drop_exact_same(user_df)
    user_df = drop_na(user_df)
    user_df = fill_hat_and_shoes_as_transparency(user_df)
    user_df = item_name_processing(user_df)
    user_df = item_name_to_index(user_df, item_df)

    return user_df, item_df


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    save_dir = "modeling/data/csv"
    save_path_inter = os.path.join(save_dir, "test_data.csv")
    save_path_item = os.path.join(save_dir, "item_data.csv")

    # 파일 저장되어 있으면 그냥 불러와서 쓰기
    if os.path.exists(save_dir):
        return pd.read_csv(save_path_inter), pd.read_csv(save_path_item)

    # 아니면 만들기
    os.makedirs(save_dir)

    file_names = [
        "csv/user_detail_eunhye.csv",
        "csv/user_detail_jeong.csv",
        "csv/user_detail_ryu.csv",
        "csv/user_detail_sssu.csv",
        "csv/user_detail_wonjun.csv",
    ]
    last_update = (2023, 2, 4, 20, 55, 49)

    dfs = []
    for file_name in file_names:
        inter, item_df = read_df_after_crawling(file_name, last_update)
        dfs.append(inter)

    inter_df = pd.concat(dfs).reset_index(drop=True)

    # 기존 학습 데이터와 동일한 row 다 날리기
    train_df = BigQueryHelper("keys//gcs_key.json").read_df_from_table("MCN")
    inter_df = inter_df[~inter_df.isin(train_df)].dropna().astype(int)

    inter_df.to_csv(save_path_inter, index=False)
    item_df.to_csv(save_path_item, index=False)

    return inter_df, item_df


def download_images(item_data: pd.DataFrame) -> torch.Tensor:
    # 1. 이미지 다운로드
    path = "modeling/data/image"
    saved_path = os.path.join(path, "item")
    if os.path.exists(saved_path):
        print("이미지가 이미 저장되어 있습니다.")
    else:
        os.makedirs(path)
        print("이미지 다운로드를 시작합니다.")
        gcs_helper.change_bucket("maple_raw_data")
        tar_file_path = os.path.join(path, "item_image.tar.gz")
        gcs_helper.download_file_from_gcs(
            blob_name="image/item_image.tar.gz", file_name=tar_file_path
        )

        with tarfile.open(tar_file_path, "r:gz") as tr:
            tr.extractall(path=path)

    # 2. Tensor로 변환
    image_tensors = [None for _ in range(len(item_data))]

    # 간혹 이미지가 오류가 나는 친구들도 존재
    # 그 경우 그냥 해당 카테고리의 평균 이미지 (dummy) 로 처리
    dummy = item_data[item_data["category"] == "dummy"]

    # 이미지 사이즈 모델에 맞게 변경해야함!
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((24, 24)),
            torchvision.transforms.ToTensor(),
        ]
    )

    print("이미지를 텐서로 변환합니다. ")
    for i, row in tqdm(item_data.iterrows()):
        image_path = row["local_image_path"]
        item_category = row["equip_category"]
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image_path = dummy[dummy["equip_category"] == item_category][
                "local_image_path"
            ].values[0]

            image = Image.open(image_path).convert("RGB")

        image_tensors[i] = trans(image)

    return torch.stack(image_tensors)
