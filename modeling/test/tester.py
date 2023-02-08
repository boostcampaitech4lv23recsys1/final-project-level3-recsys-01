import torch
from torch import nn
from torch.utils.data import DataLoader


from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from modeling.test.test_dataset import TopkDataset, ProductDataset


class Tester:
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        test_dataloader: DataLoader,
        item_data: pd.DataFrame,
        image_tensors: Optional[torch.Tensor] = None,
    ) -> None:

        # 인풋 값 저장
        self.config = config
        self.model = model
        self.test_dataloader = test_dataloader
        self.image_tensors = image_tensors

        self.device = config["device"]
        self.topk = 10

        self.item_parts, self.dummy_to_index = self.split_item_by_part(item_data)

    def test(self) -> None:
        self.model = self.model.to(self.device)
        self.model.eval()
        progress_bar = tqdm(self.test_dataloader)

        scores = 0

        # 어차피 batch size가 1이다.
        for batch_num, batch in enumerate(progress_bar, 1):

            # 현재 장비 점수 기록하기
            images = self.image_tensors[batch]
            cur_score = self.model(images.to(self.device)).item()

            # batch는 (1, 7) 7은 7개의 아이템 (batch size를 1로 설정했기 때문)
            equips = batch.squeeze()

            # 랜덤하게 3가지 부위 고르기
            # 1이면 마스킹 된 것.
            bool_tensor = torch.LongTensor([1] * 3 + [0] * 4)
            bool_tensor = bool_tensor[torch.randperm(7)]

            # 1인 부분 값을 -1로 변경
            equips = equips * (1 - bool_tensor) - bool_tensor

            # 마스킹 된 부위 및 원래 착용 안한 부위 dummy 변환
            equips_list = equips.tolist()
            equips_list = [
                self.dummy_to_index[i] if equips_list[i] == -1 else equips_list[i]
                for i in range(7)
            ]

            part_index_and_topk = self.get_topk(equips_list, bool_tensor)
            score = self.product_topk(equips_list, part_index_and_topk, cur_score)
            scores += score

            progress_bar.set_postfix_str(
                f"현재 개수: {scores} / 전체 개수 {len(self.test_dataloader)}"
            )

        return scores / len(self.test_dataloader)

    @torch.no_grad()
    def get_topk(
        self, equips_list: List[int], bool_tensor: torch.Tensor
    ) -> Dict[int, List[int]]:
        part_index_and_topk = dict()

        # 부위별 top k 탐색 시작
        for part_index, equip in enumerate(equips_list):
            part_scores = list()

            # 마스킹 된 부분만 top k 선정
            if not bool_tensor[part_index]:
                continue

            item_part = self.item_parts[part_index]

            # 특정 비율만 보기.
            dataset_per_part = TopkDataset(equips_list, part_index, item_part)
            dataloader_per_part = DataLoader(
                dataset=dataset_per_part, batch_size=32, shuffle=False
            )

            for batch_idx, batch in enumerate(dataloader_per_part):
                # batch shape (batch, 7) == (16, 7)
                images = self.image_tensors[batch]
                output = self.model(images.to(self.device))
                result = [
                    (batch[i, part_index], float(output[i]))
                    for i in range(batch.shape[0])
                ]
                part_scores.extend(result)

            part_scores.sort(key=lambda x: x[1], reverse=True)
            part_index_and_topk[part_index] = [x[0] for x in part_scores[: self.topk]]

        return part_index_and_topk

    @torch.no_grad()
    def product_topk(
        self,
        equips_list: List[int],
        part_index_and_topk: Dict[int, List[int]],
        cur_score: float,
    ) -> List[Tuple[torch.Tensor, float]]:
        # top k개에 대한 모든 경우의 수 고려
        dataset_for_product = ProductDataset(equips_list, part_index_and_topk)
        dataloader_for_product = DataLoader(
            dataset=dataset_for_product, batch_size=32, shuffle=False
        )

        total_count = 0

        for batch_idx, batch in enumerate(dataloader_for_product):
            # batch shape (batch, 7) == (16, 7)
            images = self.image_tensors[batch]
            output = self.model(images.to(self.device))
            count_per_batch = (output > cur_score).count_nonzero().item()
            total_count += count_per_batch
            if total_count > self.topk:
                return 0

        return 1

    def split_item_by_part(
        self, item_data: pd.DataFrame
    ) -> Tuple[List[int], Dict[int, int]]:
        item_parts = []
        dummy_to_index = {
            0: 10093,
            1: 10094,
            2: 10095,
            3: 10097,
            4: 10098,
            5: 10099,
            6: 10100,
        }
        item_data = item_data[item_data["is_cash"] == 1]

        for part in ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]:
            if part == "Top":
                overall = item_data[item_data["equip_category"] == "Overall"][
                    "index"
                ].tolist()
                top = item_data[item_data["equip_category"] == "Top"]["index"].tolist()
                item_by_part = overall + top
            else:
                item_by_part = item_data[item_data["equip_category"] == part][
                    "index"
                ].tolist()

            item_parts.append(item_by_part)

        return item_parts, dummy_to_index
