import pandas as pd
import os
from tqdm import tqdm


if __name__ == "__main__":
    df = pd.read_csv("data/maple_item.csv")
    save_folder = "data/maple_item_img"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for id, url in tqdm(zip(df["id"].values, df["imgUrl"].values)):
        os.system("curl " + url + f" > data/maple_item_img/{id}.png")