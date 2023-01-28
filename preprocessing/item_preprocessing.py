import pandas as pd
from utils import GCSHelper
from tqdm import tqdm
import re

gcs_helper = GCSHelper(key_path="keys/gcs_key.json", bucket_name="maple_raw_data")

# 각 버전에 대한 자세한 설명은 노션 및 ipynb 참고


def ver1_1():
    dependent_file = "item_KMST_1149_VER1.0.csv"
    drop_list = [
        "nickname",
        "codi-hat",
        "codi-hair",
        "codi-face",
        "codi-top",
        "codi-bottom",
        "codi-shoes",
        "codi-weapon",
    ]
    gcs_helper.change_bucket("maple_raw_data")
    inter_df = gcs_helper.read_df_from_gcs(
        blob_name="csv/user_total/user_detail_total.csv"
    ).drop_duplicates(drop_list)

    gcs_helper.change_bucket("maple_preprocessed_data")
    item_df = gcs_helper.read_df_from_gcs(blob_name=dependent_file)

    item_df["requiredJobs"] = item_df["requiredJobs"].replace("None", "['Beginner']")
    item_df["requiredLevel"] = item_df["requiredLevel"].replace("None", 0)

    item_drop_none = item_df[item_df["name"] != "None"]
    item_drop_none = item_drop_none[item_drop_none["name"] != "-"]

    pattern = r"[!@#$%^&*()_+-=`~,.<>/?{}\s\[\]]"
    item_drop_none["name_processed"] = item_drop_none["name"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )

    item_drop_duplicate = item_drop_none.drop_duplicates(
        ["name_processed"]
    ).reset_index(drop=True)

    item_list_from_maple_gg = [
        inter_df["codi-hat"],
        inter_df["codi-hair"],
        inter_df["codi-face"],
        inter_df["codi-top"],
        inter_df["codi-bottom"],
        inter_df["codi-shoes"],
        inter_df["codi-weapon"],
    ]
    pattern = r"[!@#$%^&*()_+-=`~,.<>/?{}\s\[\]]"
    item_list_from_maple_gg = (
        pd.concat(item_list_from_maple_gg)
        .reset_index(drop=True)
        .apply(lambda x: re.sub(pattern, "", x.lower()))
    )
    item_list_from_maple_gg = item_list_from_maple_gg[
        item_list_from_maple_gg != ""
    ].unique()

    item_set_from_maplestory_io = set(item_drop_duplicate["name_processed"].tolist())
    not_in_maplestory_io = set()
    can_use_name_directly_to_key = set()

    for item_name_from_maple_gg in item_list_from_maple_gg:
        if item_name_from_maple_gg not in item_set_from_maplestory_io:
            not_in_maplestory_io.add(item_name_from_maple_gg)
        else:
            can_use_name_directly_to_key.add(item_name_from_maple_gg)

    in_maplestory_io_use_contain = set()
    can_use_name_to_key_with_small_fix = set()

    for item_name_from_maple_gg in tqdm(not_in_maplestory_io):
        for item_name_from_maplestory_io in item_set_from_maplestory_io:

            if item_name_from_maple_gg in item_name_from_maplestory_io:
                in_maplestory_io_use_contain.add(item_name_from_maple_gg)
                can_use_name_to_key_with_small_fix.add(item_name_from_maplestory_io)

    item_drop_color = item_drop_duplicate.copy()
    pattern = r"^.{2}색"
    for item_name_from_maplestory_io in tqdm(can_use_name_to_key_with_small_fix):
        item_drop_color.loc[
            item_drop_color["name_processed"] == item_name_from_maplestory_io,
            "name_processed",
        ] = re.sub(pattern, "", item_name_from_maplestory_io)

    item_drop_color = item_drop_color.drop_duplicates(["name_processed"]).reset_index(
        drop=True
    )

    item_drop_color["requiredGender"] = item_drop_color["requiredGender"].apply(
        lambda x: min(x, 2)
    )
    return item_drop_color


def ver1_2():
    dependent_file = "item_KMST_1149_VER1.1.csv"
    gcs_helper.change_bucket("maple_preprocessed_data")
    df = gcs_helper.read_df_from_gcs(dependent_file)

    def equipCategory(x):
        if x in ["Hat", "Top", "Face", "Hair", "Overall", "Bottom", "Shoes"]:
            return x
        return "Weapon"

    df["equipCategory"] = df["subCategory"].apply(equipCategory)
    return df


def ver1_3():
    dependent_file = "item_KMST_1149_VER1.2.csv"
    gcs_helper.change_bucket("maple_preprocessed_data")
    df = gcs_helper.read_df_from_gcs(dependent_file)
    df["gcs_image_url"] = (
        "https://storage.googleapis.com/maple_web/" + df["gcs_image_url"]
    )
    return df


def ver_1_4():
    dependent_file = "item_KMST_1149_VER1.3.csv"
    gcs_helper.change_bucket("maple_preprocessed_data")
    data = gcs_helper.read_df_from_gcs(dependent_file)

    data["isCash"] = data["isCash"].apply(int)

    rename_dict = {
        k: v
        for k, v in zip(
            data.columns,
            [
                "required_jobs",
                "required_level",
                "required_gender",
                "is_cash",
                "desc",
                "item_id",
                "name",
                "overall_category",
                "category",
                "sub_category",
                "low_item_id",
                "high_item_id",
                "image_url",
                "gcs_image_url",
                "name_processed",
                "equip_category",
            ],
        )
    }
    data = data.rename(columns=rename_dict)
    return data


def ver_1_5():
    dependent_file = "item_KMST_1149_VER1.0.csv"

    gcs_helper.change_bucket("maple_preprocessed_data")
    item_df = gcs_helper.read_df_from_gcs(blob_name=dependent_file)

    item_df["requiredJobs"] = item_df["requiredJobs"].replace("None", "['Beginner']")
    item_df["requiredLevel"] = item_df["requiredLevel"].replace("None", 0)

    item_df = item_df[item_df["name"] != "None"]
    item_df = item_df[item_df["name"] != "-"]

    pattern = r"[!@#$%^&*()_+-=`~,.<>/?{}\s\[\]]"
    item_df["name_processed"] = item_df["name"].apply(
        lambda x: re.sub(pattern, "", x.lower())
    )

    pattern = r"검은색|빨간색|주황색|노란색|초록색|파란색|보라색|갈색"
    new_names = []
    for i, row in item_df.iterrows():
        if row["subCategory"] != "Hair":
            new_names.append(row["name_processed"])
        else:
            new_names.append(re.sub(pattern, "", row["name_processed"]))

    item_df["name_processed"] = new_names
    item_df = item_df.drop_duplicates(["name_processed"]).reset_index(drop=True)

    item_df["requiredGender"] = item_df["requiredGender"].apply(lambda x: min(x, 2))

    def equipCategory(x):
        if x in ["Hat", "Top", "Face", "Hair", "Overall", "Bottom", "Shoes"]:
            return x
        return "Weapon"

    item_df["equipCategory"] = item_df["subCategory"].apply(equipCategory)

    item_df["gcs_image_url"] = (
        "https://storage.googleapis.com/maple_web/" + item_df["gcs_image_url"]
    )

    item_df["isCash"] = item_df["isCash"].apply(int)

    rename_dict = {
        k: v
        for k, v in zip(
            item_df.columns,
            [
                "required_jobs",
                "required_level",
                "required_gender",
                "is_cash",
                "desc",
                "item_id",
                "name",
                "overall_category",
                "category",
                "sub_category",
                "low_item_id",
                "high_item_id",
                "image_url",
                "gcs_image_url",
                "name_processed",
                "equip_category",
            ],
        )
    }
    item_df = item_df.rename(columns=rename_dict)

    item_df["local_image_path"] = "modeling/data/" + item_df["gcs_image_url"].apply(
        lambda x: x[41:]
    )

    item_df = item_df.reset_index()

    hat_row = [
        10093,
        "dummy",
        0,
        0,
        0,
        "None",
        1,
        "dummy_hat",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Hat",
        "modeling/data/image/item/1.png",
    ]
    hair_row = [
        10094,
        "dummy",
        0,
        0,
        0,
        "None",
        2,
        "dummy_hair",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Hair",
        "modeling/data/image/item/2.png",
    ]
    face_row = [
        10095,
        "dummy",
        0,
        0,
        0,
        "None",
        3,
        "dummy_face",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Face",
        "modeling/data/image/item/3.png",
    ]
    overall_row = [
        10096,
        "dummy",
        0,
        0,
        0,
        "None",
        4,
        "dummy_overall",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Overall",
        "modeling/data/image/item/4.png",
    ]
    top_row = [
        10097,
        "dummy",
        0,
        0,
        0,
        "None",
        5,
        "dummy_top",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Top",
        "modeling/data/image/item/5.png",
    ]
    bottom_row = [
        10098,
        "dummy",
        0,
        0,
        0,
        "None",
        6,
        "dummy_bottom",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Bottom",
        "modeling/data/image/item/6.png",
    ]
    shoes_row = [
        10099,
        "dummy",
        0,
        0,
        0,
        "None",
        7,
        "dummy_sheos",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Sheos",
        "modeling/data/image/item/7.png",
    ]
    weapon_row = [
        10100,
        "dummy",
        0,
        0,
        0,
        "None",
        8,
        "dummy_weapon",
        "dummy",
        "dummy",
        "dummy",
        0,
        0,
        "dummy",
        "dummy",
        "dummy",
        "Weapon",
        "modeling/data/image/item/8.png",
    ]

    dummy = pd.DataFrame(
        [
            hat_row,
            hair_row,
            face_row,
            overall_row,
            top_row,
            bottom_row,
            shoes_row,
            weapon_row,
        ]
    )
    dummy.columns = item_df.columns

    item_df = pd.concat([item_df, dummy], axis=0)

    return item_df
