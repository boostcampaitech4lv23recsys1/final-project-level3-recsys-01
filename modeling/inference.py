from utilities import read_json, loading_text_file
from dataloader import Preprocess
from datetime import datetime
from pytz import timezone
import model as models
from torch.nn import Module
import numpy as np
from typing import List
import pandas as pd
import argparse
import torch
import os


def inference_ver1(
    model: Module,
    config: dict,
    item_data: pd.DataFrame,
    equip: list,
    fixed: list,
    item2idx: dict,
) -> List[List]:  # default 추천 방식
    print("----------------------------INFERENCE VER 1---------------------------")
    model_name = config["arch"]["type"]  # 저장된 모델 불러오기 및 eval 모드
    model_path = os.path.join(config["trainer"]["save_dir"], model_name)
    model_path = os.path.join(model_path, f"{model_name}.pt")  # 불러올 모델 선택
    load_state = torch.load(model_path)
    model.load_state_dict(load_state["state_dict"])
    model.eval()
    print("...item part listing...")
    item_part_list = item_part(config, item_data)  # 부위별 아이템 리스트 (8개)
    predict_list = []
    for f in fixed:
        part_score = []
        if f == -1:
            predict_list.append([])
            continue
        else:
            certain_part = item_part_list[f]  # 부위별 아이템 리스트를 가져옴
            for certain in certain_part:  # 부위별 전체 아이템 중 하나를 가져옴
                temp_equip = equip[:]  # 현재 착용중인 장비
                temp_equip.append(item2idx[certain])  # 착용중인 장비에 추가
                output = model(torch.tensor(temp_equip))  # 그걸 model에 넣고 유사도 측정
                part_score.append(
                    (certain, float(output))
                )  # part_score에 점수와 함께 그 아이템을 저장함
        part_score.sort(key=lambda x: x[1], reverse=True)  # 점수 기준 정렬
        part_recommendation = part_score[
            : config["inference"]["top_k"]
        ]  # top_k만큼 상위에서 자름
        predict_list.append(part_recommendation)  # predict_list에 저장
    return predict_list


def item_part(config, item_data: pd.DataFrame) -> List[np.array]:
    item_data = item_data.drop_duplicates(subset="name")
    hat_name_list = item_data[
        (item_data["subCategory"] == "Hat") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    hair_name_list = item_data[
        (item_data["subCategory"] == "Hair") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    Face_name_list = item_data[
        (item_data["subCategory"] == "Face") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    Overall_name_list = item_data[
        (item_data["subCategory"] == "Overall") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    Top_name_list = item_data[
        (item_data["subCategory"] == "Top") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    Bottom_name_list = item_data[
        (item_data["subCategory"] == "Bottom") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    Shoes_name_list = item_data[
        (item_data["subCategory"] == "Shoes") & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    weapon_name_list = item_data[
        (item_data["category"].isin(["One-Handed Weapon", "Two-Handed Weapon"]))
        & (item_data["subCategory"].isin(config["preprocess"]["item_subCategory"]))
        & (item_data["isCash"] == True)
    ]["name_processed"].unique()
    return [
        hat_name_list,
        hair_name_list,
        Face_name_list,
        Overall_name_list,
        Top_name_list,
        Bottom_name_list,
        Shoes_name_list,
        weapon_name_list,
    ]


def main(config, equipment: dict) -> None:
    print("-----------------------------START INFERENCE--------------------------")
    preprocess = Preprocess(config)
    item_data = preprocess.load_data()
    print("...load txt file...")

    # 저장된 dictionary txt 불러오기
    # idx2item = loading_text_file("idx2item")
    item2idx = loading_text_file("item2idx")
    user_item_len = loading_text_file("user_item_len")
    config["arch"]["args"]["n_users"] = user_item_len[0]
    config["arch"]["args"]["n_items"] = user_item_len[1]

    # 고정할 부위(1)인 경우 equips에 idx화 하여 넣어놓기, 고정되지 않은 부위가 추천 대상
    print("...item indexing...")
    equip_name = list(dict(equipment["cur_codi"]).values())
    equip_fixed = list(dict(equipment["fix_equip"]).values())
    equip = []
    cnt = 0
    for en in equip_name:
        if equip_fixed[cnt] == -1 and en != "-":
            equip.append(item2idx[en])
        cnt += 1

    print("...load saved model...")
    model = models.get_models(config)
    final_codi = inference_ver1(model, config, item_data, equip, equip_fixed, item2idx)
    for i, p in enumerate(final_codi):
        if not p:
            final_codi[i].append(equip_name[i])
        else:
            continue

    print("...save csv file...")
    cur_time = str(datetime.now(timezone("Asia/Seoul")))[:19]
    save_path = config["inference"]["result_dir"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(
        save_path,
        f"inference_{cur_time}.csv",
    )
    final_codi = pd.DataFrame(
        [list(map(lambda x: x[0], fc)) if len(fc) != 1 else fc for fc in final_codi]
    )
    final_codi = final_codi.reset_index()
    final_codi = final_codi.fillna(method="ffill", axis=1)
    final_codi.to_csv(save_path, index=None)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Final Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="./config/mfconfig.json",
        type=str,
        help='config 파일 경로 (default: "./config/mfconfig.json")',
    )
    args.add_argument(
        "-e",
        "--equipment",
        default="./config/inference.json",
        type=str,
        help='inference config 파일 경로 (default: "./config/mfconfig.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)
    equipment = read_json(args.equipment)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config, equipment)
