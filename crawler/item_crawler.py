import requests
import pandas as pd
import json
import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import GCS_helper


COLUMNS = [
    "requiredJobs",
    "requiredLevel",
    "requiredGender",
    "isCash",
    "desc",
    "id",
    "name",
    "overallCategory",
    "category",
    "subCategory",
    "lowItemId",
    "highItemId",
]

MAIN_KEYS = [
    "requiredJobs",
    "requiredLevel",
    "requiredGender",
    "isCash",
    "desc",
    "id",
    "name"
]

TYPE_INFO_KEYS = [
    "overallCategory",
    "category",
    "subCategory",
    "lowItemId",
    "highItemId",
]


def item_info_crawing(
        client="KMST",
        version=1149,
):
    req = requests.get(f"https://maplestory.io/api/{client}/{version}/item")
    jsons = json.loads(req.text)

    df = pd.DataFrame(columns=COLUMNS)
    for idx, item_json in enumerate(tqdm(jsons, desc="maple item crawling")):
        item = list()
        for key in MAIN_KEYS:
            try:
                value = item_json[key]
                if value == "":
                    value = "None"

                item.append(value)
            except:
                item.append("None")

        for key in TYPE_INFO_KEYS:
            try:
                value = item_json["typeInfo"][key]
                if value == "":
                    value = "None"

                item.append(value)
            except:
                item.append("None")
        df.loc[idx] = item

    df["image_url"] = df["id"].apply(lambda x: f"https://maplestory.io/api/KMST/1149/item/{x}/icon")

    return df


def item_image_crawling(
        item_df: pd.DataFrame,
        gcs_helper: GCS_helper,
):
    for row in tqdm(item_df.iterrows(), total=len(item_df)):
        item_id = row[1]["id"]
        item_url = row[1]["image_url"]
        gcs_path = f"image/item/{item_id}.png"

        if not gcs_helper.path_exists(path=gcs_path):
            gcs_helper.upload_image_to_gcs(
                blob_name=gcs_path,
                image_url=item_url
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--client', type=str, default='KMST')
    arg('--version', type=str, default='1149')
    arg('--key_path', type=str, default='../keys/gcs_key.json')
    args = parser.parse_args()

    gcs_helper = GCS_helper(args.key_path)
    save_file_path = f"csv/maple_item_{args.client}_{args.version}.csv"

    if not gcs_helper.path_exists(path=save_file_path):
        print("---------ITEM INFO CRAWLING START---------")
        item_df = item_info_crawing(
            client=args.client,
            version=args.version
        )
        print("---------ITEM INFO CRAWLING FINISH---------")
        print("---------ITEM INFO DATAFRAME SAVE START---------")
        gcs_helper.upload_df_to_gcs(
            blob_name=save_file_path,
            df=item_df
        )
        print("---------ITEM INFO DATAFRAME SAVE FINISH---------")
    else:
        print("---------FILE ALREADY EXIST---------")
        item_df = gcs_helper.read_df_from_gcs(blob_name=save_file_path)


    print("---------ITEM IMAGE CRAWLING AND SAVE START---------")
    item_image_crawling(
        item_df=item_df,
        gcs_helper=gcs_helper
    )
    print("---------ITEM IMAGE CRAWLING AND SAVE FINISH---------")
