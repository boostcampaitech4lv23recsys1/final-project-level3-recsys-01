import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import GCSHelper
import re
import os


def get_int_basic(data: pd.DataFrame) -> pd.DataFrame:
    """_Modify string columns as integer
    (level, exp, popularity, total_ranking, world_ranking, class_wolrd_ranking, class_total_ranking)

    Args:
        data (pd.DataFrame): Base dataframe

    Returns:
        pd.DataFrame: Reformatted dataframe
    """
    data["exp"] = data["level"].apply(lambda x: float(x.split("(")[-1][:-2]))

    data["level"] = data["level"].apply(
        lambda x: int(re.search("\d+\(", x).group()[:-1])
    )
    data["popularity"] = data["popularity"].apply(
        lambda x: int(x.split("\n")[1].replace(",", ""))
    )
    data["total_ranking"] = data["total_ranking"].apply(
        lambda x: int(x.replace("위", "").replace(",", ""))
    )
    data["world_ranking"] = data["world_ranking"].str.replace('-', '-1')
    data["world_ranking"] = data["world_ranking"].apply(
        lambda x: int(x.replace("위", "").replace(",", ""))
    )
    data["class_world_ranking"] = data["class_world_ranking"].str.replace('-', '-1')
    data["class_world_ranking"] = data["class_world_ranking"].apply(
        lambda x: int(x.replace("(월드)", "").replace("위", "").replace(",", ""))
    )
    data["class_total_ranking"] = data["class_total_ranking"].str.replace('-', '-1')
    data["class_total_ranking"] = data["class_total_ranking"].apply(
        lambda x: int(x.replace("(전체)", "").replace("위", "").replace(",", ""))
    )
    return data


def get_int_record(data: pd.DataFrame) -> pd.DataFrame:
    """Modify Null into -1
    (mureung, theseed, union, achievement)

    Args:
        data (pd.DataFrame): Base dataframe

    Returns:
        pd.DataFrame: Reformatted dataframe
    """
    data["mureung"] = data["mureung"].apply(
        lambda x: -1 if x == "기록이 없습니다." else int(x.replace(",", ""))
    )
    data["theseed"] = data["theseed"].apply(
        lambda x: -1 if x == "기록이 없습니다." else int(x.replace(",", ""))
    )
    data["union"] = data["union"].apply(
        lambda x: -1 if x == "기록이 없습니다." else int(x.replace(",", ""))
    )
    data["achievement"] = data["achievement"].apply(
        lambda x: -1 if x == "기록이 없습니다." else int(x.replace(",", ""))
    )
    return data


def date_form_fix(data: pd.DataFrame) -> pd.DataFrame:
    data["last_access"] = data["last_access"].apply(
        lambda x: x.replace("/", "_").replace("-", "_")
    )
    return data

def drop_na(data: pd.DataFrame) -> pd.DataFrame:
    drop_index = data[(data['codi-hat']=='-')&(data['codi-hair']=='-')&(data['codi-face']=='-')&(data['codi-top']=='-')&(data['codi-bottom']=='-')&(data['codi-shoes']=='-')&(data['codi-weapon']=='-')].index
    data = data.drop(drop_index, axis=0).reset_index(drop=True)
    return data

def fill_hat_and_shoes_as_transparency(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[data['codi-hat']=='-','codi-hat'] = '투명모자'
    data.loc[data['codi-shoes']=='-','codi-shoes'] = '투명신발'
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="기본적인 전처리를 진행합니다.")
    parser.add_argument(
        "--data_dir",
        default="final-project-level3-recsys-01/data/",
        required=True,
        help="데이터의 저장 위치 및 파일 명 까지 입력해주세요.",
    )
    parser.add_argument(
        "--save_dir",
        default="final-project-level3-recsys-01/data/",
        required=True,
        help="전처리 된 데이터의 저장 위치만 입력해주세요."
    )
    parser.add_argument(
        "--upload_gcs",
        default=False,
        help="gcs에 올릴지 여부를 입력해주세요. 기본값은 False입니다."
    )
    parser.add_argument(
        "--key_path",
        default='./keys/gcs_key.json',
        help="gcs 키 경로를 입력해주세요."
    )
    parser.add_argument(
        "--version",
        required=False,
        help="gcs에 올라갈 전처리 버전을 입력해주세요."
    )

    args = parser.parse_args()

    data = pd.read_csv(args.data_dir)
    data = get_int_basic(data)
    data = get_int_record(data)
    data = date_form_fix(data)
    data = drop_na(data)
    data = fill_hat_and_shoes_as_transparency(data)
    data.to_csv(
        os.path.join(args.save_dir, "basic_preprocessed_data.csv"),
        encoding="utf-8-sig",
        index=False,
    )
    if args.upload_gcs :
        gcs_helper = GCSHelper(args.key_path, bucket_name = "maple_preprocessed_data")
        gcs_helper.upload_df_to_gcs(f"user_detail_VER{args.version}.csv", data)
