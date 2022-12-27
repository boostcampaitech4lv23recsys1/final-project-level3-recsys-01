import pandas as pd
import os


if __name__ == "__main__":
    user_info_df = pd.DataFrame(columns=["server", "user_info_url", "user"])

    base_path = "user_info"
    for user_info_path in os.listdir(base_path):
        temp = pd.read_csv(os.path.join(base_path, user_info_path))
        user_info_df = pd.concat([user_info_df, temp])

    user_info_df.columns = ["server", "user_info_url", "nickname"]

    user_detail_df = pd.DataFrame(columns=['nickname', 'codi-hat', 'codi-hair', 'codi-face', 'codi-top',
                                           'codi-bottom', 'codi-shoes', 'codi-weapon', 'level', 'class',
                                           'popularity', 'total_ranking', 'world_ranking', 'class_world_ranking',
                                           'class_total_ranking', 'guild', 'last_access', 'mureung', 'theseed',
                                           'union', 'achievement', 'cur_chr', 'past_chr_img_1', 'past_chr_img_2',
                                           'past_chr_img_3', 'past_chr_img_4', 'past_chr_img_5', 'past_chr_img_6',
                                           'past_chr_date_1', 'past_chr_date_2', 'past_chr_date_3',
                                           'past_chr_date_4', 'past_chr_date_5', 'past_chr_date_6'])

    base_path = "user_detail"
    for user_detail_path in os.listdir(base_path):
        temp = pd.read_csv(os.path.join(base_path, user_detail_path))
        user_detail_df = pd.concat([user_detail_df, temp])

    user_detail_df = user_detail_df.drop_duplicates(subset=["nickname"])

    user_df = pd.merge(left=user_detail_df, right=user_info_df, on="nickname", how="left")

    user_df.to_csv("user_data.csv", index=False)
