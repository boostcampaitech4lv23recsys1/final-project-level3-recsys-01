import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import os
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from typing import Dict, Any


from modeling.trainer.loss import get_loss
from modeling.trainer.optimizer import get_optimizer
from modeling.utils.gcs_helper import GCSHelper

import wandb


class SimpleMCNTrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> None:

        # 인풋 값 저장
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # loss랑 optimizer 가져오기
        self.criterion = get_loss(config["trainer"])
        self.optimizer = get_optimizer(self.model, config["trainer"]["optimizer"])

        # 학습에 필요한 기타 값들 config에서 저장해놓기
        self.device = config["device"]
        self.epochs = config["trainer"]["epochs"]

        # 최적의 모델 저장 설정
        self.best_score = 0
        self.model_name = "SimpleMCN"
        self.gcs_helper = GCSHelper(
            key_path="keys/gcs_key.json",
            bucket_name="maple_trained_model",
        )

    def train(self) -> None:
        self.model = self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            valid_score = self._valid_epoch(epoch)

            if valid_score > self.best_score:
                self.best_score = valid_score
                self._save_checkpoint()

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        progress_bar = tqdm(self.train_dataloader, desc=f"{epoch}번째 Train epoch")

        for batch_num, batch in enumerate(progress_bar, 1):
            # 로더에서 값 가져와서 GPU에 올리기
            images, target = batch
            images = images.to(self.device)
            target = target.float().to(self.device)

            # output으로 로스 계산
            output = self.model.forward(images)
            loss = self.criterion(output, target)

            # back prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 궁금하니까 로스 출력
            progress_bar.set_postfix_str(f"loss: {loss.item()}")

            log = {"train/total_loss": loss.item(), "batch_num": batch_num}
            wandb.log(log)

    @torch.no_grad()
    def _valid_epoch(self, epoch: int) -> float:
        print(f"{epoch} 번째 Valid를 시작합니다")
        self.model.eval()

        outputs = []
        for batch_num, batch in enumerate(self.valid_dataloader, 1):
            # 로더에서 값 가져와서 GPU에 올리기
            images, target = batch
            images = images.to(self.device)
            target = target.float().to(self.device)

            # batch 당 output 모으기
            output = self.model.forward(images)
            outputs.append(output)

        outputs = torch.cat(outputs).detach().cpu().numpy()
        valid_score = np.mean(outputs)

        print(f"{epoch} 번째 Valid 결과 {valid_score}점 입니다. ")
        log = {"valid/avg_score": valid_score, "valid_step": epoch}
        wandb.log(log)

        return valid_score

    def _save_checkpoint(self) -> str:
        print("모델 저장을 시작합니다.")

        now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y%m%d-%H%M")
        model_save_path = (
            self.config["trainer"]["save_dir"] + f"/{self.model_name}_{now}.pt"
        )
        torch.save(self.model.state_dict(), model_save_path)
        print("Saved best model to {}".format(model_save_path))

        self.gcs_helper.upload_model_to_gcs(
            f"{self.model_name}/{self.model_name}_{now}.pt", model_save_path
        )
