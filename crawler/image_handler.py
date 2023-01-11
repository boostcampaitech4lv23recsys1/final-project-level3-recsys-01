import pandas as pd
import os
from tqdm import tqdm
import argparse


def drop_past_add_column(user_detail_info: pd.DataFrame) -> pd.DataFrame:
    """
    더이상 이용하지 않는 열들을 제거하고 gcs image path 열을 추가합니다.
    """

    user_detail_info.drop(
        [
            'past_chr_img_1',
            'past_chr_img_2',
            'past_chr_img_3',
            'past_chr_img_4',
            'past_chr_img_5',
            'past_chr_img_6',
            'past_chr_date_1',
            'past_chr_date_2',
            'past_chr_date_3',
            'past_chr_date_4',
            'past_chr_date_5',
            'past_chr_date_6',
        ],
        axis=1,
        inplace=True,
    )

    cur_date = sorted(user_detail_info["last_access"], reverse=True)[0].replace(
        "/", "_"
    )

    # {경로}_{닉네임}_{저장날짜} 형식
    user_detail_info["gcs_image_path"] = user_detail_info["nickname"].apply(
        lambda x: "image/user/" + x + "/" + x + "_" + cur_date + ".png"
    )

    return user_detail_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--base_folder', type=str, default='../data/user_detail')
    args = parser.parse_args()

    for csv_file in tqdm(os.listdir(args.base_folder)):
        if not csv_file.endswith(".csv"):
            continue
        csv_path = os.path.join(args.base_folder, csv_file)

        data = pd.read_csv(csv_path)
        data = drop_past_add_column(data)
        data.to_csv(csv_path, index=False)