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
    사용하는 컬럼만 놔두고 나머진 버리기
    """
    use_columns = [
        "nickname",
        "codi-hat",
        "codi-hair",
        "codi-face",
        "codi-top",
        "codi-bottom",
        "codi-shoes",
        "codi-weapon",
    ]
    return data[use_columns]


def drop_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    모든 데이터가 다 - 인 경우 버리기
    """
    drop_index = data[
        (data["codi-hat"] == "-")
        & (data["codi-hair"] == "-")
        & (data["codi-face"] == "-")
        & (data["codi-top"] == "-")
        & (data["codi-bottom"] == "-")
        & (data["codi-shoes"] == "-")
        & (data["codi-weapon"] == "-")
    ].index
    data = data.drop(drop_index, axis=0).reset_index(drop=True)
    return data


def fill_hat_and_shoes_as_transparency(data: pd.DataFrame) -> pd.DataFrame:
    """
    모자랑 신발이 없는 경우는 투명이라고 생각하기
    """
    data.loc[data["codi-hat"] == "-", "codi-hat"] = "투명모자"
    data.loc[data["codi-shoes"] == "-", "codi-shoes"] = "투명신발"
    return data


def drop_exact_same(data: pd.DataFrame) -> pd.DataFrame:
    """
    닉네임, 장비가 모두 다 동일한 경우는 하나만 고려
    """
    return data.drop_duplicates().reset_index(drop=True)


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
    name2idx = item.reset_index().set_index("name_processed")["index"].to_dict()

    total_item = []

    for _, row in tqdm(user.iterrows()):
        temp_item = [-1] * 7
        is_matched = True
        for i, codi_item in enumerate(row[1:]):  # 닉네임 제외
            if codi_item == "":
                continue
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


def index_to_image_url(user_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    index2url = item_df["gcs_image_url"].to_dict()
    index2url[-1] = "이건 경로 어떡하지"
    columns = [
        "codi_hat",
        "codi_hair",
        "codi_face",
        "codi_top",
        "codi_bottom",
        "codi_shoes",
        "codi_weapon",
    ]
    for column in columns:
        user_df[column] = user_df[column].map(index2url)

    image_columns = [
        "hat_image_url",
        "hair_image_url",
        "face_image_url",
        "top_image_url",
        "bottom_image_url",
        "shoes_image_url",
        "weapon_image_url",
    ]

    user_df.columns = image_columns

    return user_df


def main():
    key_path = "keys/gcs_key.json"

    gcs_helper_preprocessed = GCSHelper(
        key_path=key_path, bucket_name="maple_preprocessed_data"
    )
    gcs_helper_raw = GCSHelper(key_path=key_path, bucket_name="maple_raw_data")

    item_df = gcs_helper_preprocessed.read_df_from_gcs("item_KMST_1149_latest.csv")
    user_df = gcs_helper_raw.read_df_from_gcs("csv/user_detail_jeong.csv")

    user_df_drop_columns = drop_columns(user_df)
    user_df_drop_na = drop_na(user_df_drop_columns)
    user_df_fill_trans = fill_hat_and_shoes_as_transparency(user_df_drop_na)
    user_df_drop_exact_same = drop_exact_same(user_df_fill_trans)
    user_df_item_name_processing = item_name_processing(user_df_drop_exact_same)
    user_df_item_name_to_index = item_name_to_index(
        user_df_item_name_processing, item_df
    )
    user_df_index_to_image_url = index_to_image_url(user_df_item_name_to_index, item_df)

    return user_df_index_to_image_url


if __name__ == "__main__":
    df = main()
