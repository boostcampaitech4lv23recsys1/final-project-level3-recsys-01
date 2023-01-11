import pandas as pd
import argparse
import re
import os


def get_int_basic(data: pd.DataFrame) -> pd.DataFrame:
    """_Modify string columns as integer
    (level, popularity, total_ranking, world_ranking, class_wolrd_ranking, class_total_ranking)

    Args:
        data (pd.DataFrame): Base dataframe

    Returns:
        pd.DataFrame: Reformatted dataframe
    """
    data["level"] = data["level"].apply(
        lambda x: int(re.search("\d+\(", x).group()[:-1])
    )
    data["popularity"] = data["popularity"].apply(
        lambda x: int(x.split("\n")[1].replace(",", ""))
    )
    data["total_ranking"] = data["total_ranking"].apply(
        lambda x: int(x.replace("위", "").replace(",", ""))
    )
    data["world_ranking"] = data["world_ranking"].apply(
        lambda x: int(x.replace("위", "").replace(",", ""))
    )
    data["class_world_ranking"] = data["class_world_ranking"].apply(
        lambda x: int(x.replace("(월드)", "").replace("위", "").replace(",", ""))
    )
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
        help="전처리 된 데이터의 저장 위치만 입력해주세요.",
    )

    args = parser.parse_args()

    data = pd.read_csv(args.data_dir)
    data = get_int_basic(data)
    data = get_int_record(data)
    data = date_form_fix(data)
    data.to_csv(
        os.path.join(args.save_dir, "basic_preprocessed_data.csv"),
        encoding="utf-8-sig",
        index=False,
    )
