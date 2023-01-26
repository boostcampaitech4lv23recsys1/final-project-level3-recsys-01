import torch
from torch.nn import Module

import pandas as pd

from datetime import datetime
from pytz import timezone
import argparse
import os
from typing import List, Dict, Tuple, Any

import modeling.model as models
from modeling.dataloader import Preprocess
from modeling.utilities import read_json


def inference_ver1(
    model: Module, config: Dict[str, Any], item_data: pd.DataFrame, equip: List[int]
) -> List[List[Tuple[int, float]]]:  # default 추천 방식
    print("----------------------------INFERENCE VER 1---------------------------")
    model_name = config["arch"]["type"]  # 저장된 모델 불러오기 및 eval 모드
    model_path = os.path.join(config["trainer"]["save_dir"], model_name)
    model_path = os.path.join(model_path, f"{model_name}.pt")  # 불러올 모델 선택
    load_state = torch.load(model_path)
    model.load_state_dict(load_state["state_dict"])
    model.eval()
    print("...item part listing...")

    item_part_list = item_part(item_data)  # 부위별 아이템 리스트 (8개)
    predict_list = []
    for part_index, item_index in enumerate(equip):
        if item_index != -1:
            predict_list.append(
                [(item_index, 1) for _ in range(config["inference"]["top_k"])]
            )
        else:
            part_score = []

            certain_part = item_part_list[part_index]  # 부위별 아이템 리스트를 가져옴
            for any_item_in_part in certain_part:  # 부위별 전체 아이템 중 하나를 가져옴
                temp_equip = equip[:]  # 현재 착용중인 장비

                # 착용중인 장비에 추가
                # 한벌옷이 학습 데이터에 따로 없어서 인덱스로 처리 해줌
                if part_index < 4:
                    temp_equip[part_index] = any_item_in_part
                else:
                    temp_equip[part_index - 1] = any_item_in_part

                output = model(torch.tensor([temp_equip]))  # 그걸 model에 넣고 유사도 측정

                part_score.append(
                    (any_item_in_part, float(output))
                )  # part_score에 점수와 함께 그 아이템을 저장함
            part_score.sort(key=lambda x: x[1], reverse=True)  # 점수 기준 정렬
            part_recommendation = part_score[
                : config["inference"]["top_k"]
            ]  # top_k만큼 상위에서 자름
            predict_list.append(part_recommendation)  # predict_list에 저장
    return predict_list


def item_part(item_data: pd.DataFrame) -> List[pd.Index]:
    hat_index_list = item_data[
        (item_data["equipCategory"] == "Hat") & (item_data["isCash"] == True)
    ].index
    hair_index_list = item_data[
        (item_data["equipCategory"] == "Hair") & (item_data["isCash"] == True)
    ].index
    Face_index_list = item_data[
        (item_data["equipCategory"] == "Face") & (item_data["isCash"] == True)
    ].index
    Overall_index_list = item_data[
        (item_data["equipCategory"] == "Overall") & (item_data["isCash"] == True)
    ].index
    Top_index_list = item_data[
        (item_data["equipCategory"] == "Top") & (item_data["isCash"] == True)
    ].index
    Bottom_index_list = item_data[
        (item_data["equipCategory"] == "Bottom") & (item_data["isCash"] == True)
    ].index
    Shoes_index_list = item_data[
        (item_data["equipCategory"] == "Shoes") & (item_data["isCash"] == True)
    ].index
    weapon_index_list = item_data[
        (item_data["equipCategory"] == "Weapon") & (item_data["isCash"] == True)
    ].index
    return [
        hat_index_list,
        hair_index_list,
        Face_index_list,
        Overall_index_list,
        Top_index_list,
        Bottom_index_list,
        Shoes_index_list,
        weapon_index_list,
    ]


def main(config: Dict[str, Any], equipment: Dict[str, Any]) -> None:
    print("-----------------------------START INFERENCE--------------------------")
    preprocess = Preprocess(config)
    item_data = preprocess.load_data()
    print("...load txt file...")

    config["arch"]["args"]["n_items"] = item_data.shape[0]

    # 고정할 부위(1)인 경우 equips에 idx화 하여 넣어놓기, 고정되지 않은 부위가 추천 대상
    print("...item indexing...")
    equip_list = list(equipment["cur_codi"].values())

    print("...load saved model...")
    model = models.get_models(config)
    final_codi = inference_ver1(model, config, item_data, equip_list)

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
        [list(map(lambda x: x[0], fc)) for fc in final_codi]
    ).transpose()
    final_codi.columns = [
        "Hat",
        "Hair",
        "Face",
        "Overall",
        "Top",
        "Bottom",
        "Shoes",
        "Weapon",
    ]
    final_codi.to_csv(save_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Final Dinosaur")
    args.add_argument(
        "-c",
        "--config",
        default="modeling/config/mfconfig.json",
        type=str,
        help='config 파일 경로 (default: "modeling/config/mfconfig.json")',
    )
    args.add_argument(
        "-e",
        "--equipment",
        default="modeling/config/inference.json",
        type=str,
        help='inference config 파일 경로 (default: "modeling/config/mfconfig.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)
    equipment = read_json(args.equipment)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config, equipment)
