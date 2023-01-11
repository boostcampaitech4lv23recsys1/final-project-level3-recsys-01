import pandas as pd
import os


def drop_past_add_column(user_detail_info: pd.DataFrame) -> pd.DataFrame:
    """

    """

    user_detail_info.drop(['past_chr_img_1', 'past_chr_img_2',
       'past_chr_img_3', 'past_chr_img_4', 'past_chr_img_5', 'past_chr_img_6',
       'past_chr_date_1', 'past_chr_date_2', 'past_chr_date_3',
       'past_chr_date_4', 'past_chr_date_5', 'past_chr_date_6'], axis=1, inplace=True)

    cur_date = sorted(user_detail_info["last_access"], reverse=True)[0].replace("/", "_")

    # {경로}_{닉네임}_{저장날짜} 형식
    user_detail_info["gcs_image_path"] = user_detail_info["nickname"].apply(lambda x:
        "image/user/" + x + "/" + x + "_" + cur_date + ".png"
    )

    return user_detail_info


if __name__ == "__main__":
    base_folder = "../data/user_detail"

    for csv_file in os.listdir(base_folder):
        if not csv_file.endswith(".csv"):
            continue
        csv_path = os.path.join(base_folder, csv_file)

        data = pd.read_csv(csv_path)
        data = drop_past_add_column(data)
        data.to_csv(csv_path, index=False)