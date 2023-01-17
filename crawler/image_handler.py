import pandas as pd
import os
from tqdm import tqdm
import argparse
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import GCSHelper


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

    cur_date = (
        sorted(user_detail_info["last_access"], reverse=True)[0]
        .replace("/", "_")
        .replace('-', '_')
    )

    # {경로}_{닉네임}_{저장날짜} 형식
    user_detail_info["gcs_image_path"] = user_detail_info["nickname"].apply(
        lambda x: "image/user/" + x + "/" + x + "_" + cur_date + ".png"
    )

    return user_detail_info


def image_uploader(
    user_detail_info: pd.DataFrame,
    gcs_helper: GCSHelper,
    img_folder: str,
    csv_folder: str,
) -> None:
    local_images = user_detail_info['gcs_image_path'].apply(
        lambda x: f"user_{x.split('/')[3]}"
    )
    gcs_path = user_detail_info['gcs_image_path']
    for image, path in tqdm(zip(local_images, gcs_path)):
        gcs_helper.upload_file_to_gcs(path, os.path.join(img_folder, csv_folder, image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--detail_folder', type=str, default='./data/user_detail')
    arg('--img_folder', type=str, default='./data/user_img')
    arg('--key_path', type=str, default='./keys/gcs_key.json')
    args = parser.parse_args()
    gcs_helper = GCSHelper(args.key_path)
    for csv_file in os.listdir(args.detail_folder):
        if not csv_file.endswith(".csv"):
            continue
        csv_path = os.path.join(args.detail_folder, csv_file)

        data = pd.read_csv(csv_path)
        try:
            data = drop_past_add_column(data)
            data.to_csv(csv_path, index=False)
        except:
            pass

        image_uploader(data, gcs_helper, args.img_folder, csv_path.split('/')[-1][:-4])
