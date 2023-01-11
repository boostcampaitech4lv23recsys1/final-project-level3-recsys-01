import pandas as pd
import os
import argparse


def making_df(base_path, base_df):
    user_df = base_df.copy()
    for path in os.listdir(base_path):
        temp = pd.read_csv(os.path.join(base_path, path))
        user_df = pd.concat([user_df, temp])
    return user_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--user_info_path", type=str, default="user_info")
    arg("--user_detail_path", type=str, default="user_detail")
    arg("--save_path", type=str, default="./")
    args = parser.parse_args()

    base_user_info = pd.DataFrame(columns=["server", "user_info_url", "user"])
    user_info_df = making_df(args.user_info_path, base_user_info)
    user_info_df.columns = ["server", "user_info_url", "nickname"]

    base_user_detail = pd.DataFrame(
        columns=[
            'nickname',
            'codi-hat',
            'codi-hair',
            'codi-face',
            'codi-top',
            'codi-bottom',
            'codi-shoes',
            'codi-weapon',
            'level',
            'class',
            'popularity',
            'total_ranking',
            'world_ranking',
            'class_world_ranking',
            'class_total_ranking',
            'guild',
            'last_access',
            'mureung',
            'theseed',
            'union',
            'achievement',
            'cur_chr',
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
        ]
    )

    user_detail_df = making_df(args.user_detail_path, base_user_detail)
    user_detail_df = user_detail_df.drop_duplicates(subset=["nickname"])

    user_df = pd.merge(
        left=user_detail_df, right=user_info_df, on="nickname", how="left"
    )

    user_df.to_csv(os.path.join(args.save_path, "user_data.csv"), index=False)
