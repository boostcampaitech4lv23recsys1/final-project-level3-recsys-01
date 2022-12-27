import os
import datetime
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    base_folder = "user_detail"
    for path in os.listdir(base_folder):
        file_path = os.path.join(base_folder, path)
        df = pd.read_csv(file_path)
        cur_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d')[2:]

        for idx in tqdm(range(len(df))):
            user_detail = df.iloc[idx]
            os.system("curl " + user_detail["cur_chr"] + f" > data/maple_user_img/user_{user_detail['nickname']}_{cur_date}.png")
            for i in range(1, 7):
                date = user_detail[f'past_chr_date_{i}'].replace("/", "_")
                os.system("curl " + user_detail[f"past_chr_img_{i}"] + f" > data/maple_user_img/user_{user_detail['nickname']}_{date}.png")