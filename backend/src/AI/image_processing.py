from torch import Tensor, stack
import torchvision
from PIL import Image
from pandas import DataFrame

import os
import tarfile
from tqdm import tqdm
from typing import Tuple, List

from src.utils import GCSHelper

from src.database.models.crud_item import find_all
from src.database.init_db import get_db


gcs_helper = GCSHelper(key_path="src/utils/gcs_key.json", bucket_name="maple_raw_data")


async def image_to_tensor() -> Tuple[Tensor, DataFrame]:
    # 1. 이미지 다운로드
    path = "src/data/image"
    saved_path = os.path.join(path, "item")
    if os.path.exists(saved_path):
        print("이미지가 이미 저장되어 있습니다.")
    else:
        os.makedirs(path)
        print("이미지 다운로드를 시작합니다.")
        gcs_helper.change_bucket("maple_raw_data")
        tar_file_path = os.path.join(path, "item_image.tar.gz")
        gcs_helper.download_file_from_gcs(
            blob_name="image/item_image.tar.gz", file_name=tar_file_path
        )

        with tarfile.open(tar_file_path, "r:gz") as tr:
            tr.extractall(path=path)
    # 2. 이미지 텐서로 변환
    print("이미지를 텐서로 변환합니다. ")
    gcs_helper.change_bucket("maple_preprocessed_data")
    db = await get_db().__anext__()
    item_data = DataFrame(await find_all(db))
    image_tensors = [None for _ in range(len(item_data))]

    # 간혹 이미지가 오류가 나는 친구들도 존재
    # 그 경우 그냥 해당 카테고리의 평균 이미지 (dummy) 로 처리
    dummy = item_data[item_data["category"] == "dummy"]

    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    for i, row in tqdm(item_data.iterrows()):
        image_path = os.path.join("src", row["local_image_path"][9:])
        item_category = row["equip_category"]
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image_path = os.path.join(
                "src",
                dummy[dummy["equip_category"] == item_category][
                    "local_image_path"
                ].values[0][9:],
            )
            image = Image.open(image_path).convert("RGB")

        image_tensors[i] = trans(image)

    return stack(image_tensors), item_data
