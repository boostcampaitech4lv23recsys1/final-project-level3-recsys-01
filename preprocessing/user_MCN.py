"""
상위 폴더에서
$ python3 -m preprocessing.user_MCN
이걸로 실행해주세요.
"""

import pandas as pd
from utils import GCSHelper
import re
from tqdm import tqdm


def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    사용하는 컬럼만 놔두고 나머진 버리기. nickname도 이젠 고려 안함
    """
    use_columns = [
        "codi-hat",
        "codi-hair",
        "codi-face",
        "codi-top",
        "codi-bottom",
        "codi-shoes",
        "codi-weapon",
    ]
    return data[use_columns]


def drop_exact_same(data: pd.DataFrame) -> pd.DataFrame:
    """
    장비가 모두 다 동일한 경우는 하나만 고려
    """
    return data.drop_duplicates().reset_index(drop=True)


def drop_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    4개 이상 - 인 경우 버리기
    """
    data["count"] = data.apply(lambda x: sum(x == "-"), axis=1)

    data = data[data["count"] <= 4].drop(["count"], axis=1).reset_index(drop=True)
    return data


def fill_hat_and_shoes_as_transparency(data: pd.DataFrame) -> pd.DataFrame:
    """
    모자랑 신발이 없는 경우는 투명이라고 생각하기
    """
    data.loc[data["codi-hat"] == "-", "codi-hat"] = "투명모자"
    data.loc[data["codi-shoes"] == "-", "codi-shoes"] = "투명신발"
    return data


def item_name_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    아이템 이름에 있는 특수문자 및 공백 제거하기, 소문자로 변환하기
    """
    pattern = r"[!@#$%^&*()_+-=`~,.<>/?{}\s\[\]]"
    data["codi-hat"] = data["codi-hat"].apply(lambda x: re.sub(pattern, "", x.lower()))
    data["codi-hair"] = data["codi-hair"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )
    data["codi-face"] = data["codi-face"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )
    data["codi-top"] = data["codi-top"].apply(lambda x: re.sub(pattern, "", x.lower()))
    data["codi-bottom"] = data["codi-bottom"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )
    data["codi-shoes"] = data["codi-shoes"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )
    data["codi-weapon"] = data["codi-weapon"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )
    return data


def item_name_to_index(user: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame:
    """
    user detail에 있는 아이템 이름 (maple.gg) item에 있는 index로 변환하기 (maplestory.io)
    """
    name2idx = item.set_index("name_processed")["index"].to_dict()

    dummy_hat_index = item[item["name"] == "dummy_hat"]["index"].values[0]
    dummy_hair_index = item[item["name"] == "dummy_hair"]["index"].values[0]
    dummy_face_index = item[item["name"] == "dummy_face"]["index"].values[0]
    dummy_top_index = item[item["name"] == "dummy_top"]["index"].values[0]
    dummy_bottom_index = item[item["name"] == "dummy_bottom"]["index"].values[0]
    dummy_shoes_index = item[item["name"] == "dummy_shoes"]["index"].values[0]
    dummy_weapon_index = item[item["name"] == "dummy_weapon"]["index"].values[0]

    dummy_index = [
        dummy_hat_index,
        dummy_hair_index,
        dummy_face_index,
        dummy_top_index,
        dummy_bottom_index,
        dummy_shoes_index,
        dummy_weapon_index,
    ]

    total_item = []

    for _, row in tqdm(user.iterrows()):
        temp_item = [-1] * 7
        is_matched = True
        for i, codi_item in enumerate(row):
            if codi_item == "":  # 착용하지 않은 경우 더미로 맵핑
                temp_item[i] = dummy_index[i]
            else:  # 착용 한 경우
                if codi_item not in name2idx:
                    is_matched = False
                    break
                temp_item[i] = name2idx[codi_item]

        if is_matched:
            total_item.append(temp_item)

    return pd.DataFrame(
        total_item,
        columns=[
            "codi_hat",
            "codi_hair",
            "codi_face",
            "codi_top",
            "codi_bottom",
            "codi_shoes",
            "codi_weapon",
        ],
    )


def main():
    key_path = "keys/gcs_key.json"

    gcs_helper_preprocessed = GCSHelper(
        key_path=key_path, bucket_name="maple_preprocessed_data"
    )
    gcs_helper_raw = GCSHelper(key_path=key_path, bucket_name="maple_raw_data")

    item_df = gcs_helper_preprocessed.read_df_from_gcs("item_KMST_1149_latest.csv")
    user_df = gcs_helper_raw.read_df_from_gcs("csv/user_detail_jeong.csv")

    user_df = drop_columns(user_df)
    user_df = drop_exact_same(user_df)
    user_df = drop_na(user_df)
    user_df = fill_hat_and_shoes_as_transparency(user_df)
    user_df = item_name_processing(user_df)
    user_df = item_name_to_index(user_df, item_df)

    return user_df


if __name__ == "__main__":
    df = main()
