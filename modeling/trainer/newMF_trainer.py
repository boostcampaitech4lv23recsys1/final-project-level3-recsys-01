import torch
import sys
import os
from tqdm import tqdm
from datetime import datetime
from pytz import timezone

# to import ../../utils.py
sys.path.append(
    os.path.dirname(
        os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    )
)
from utils import GCSHelper


class newMFTrainer:
    def __init__(self, config, model, train_data_loader, valid_data_loader):
        self.model = model

        self.config = config
        self.cfg_trainer = config["trainer"]
        self.epoch = self.cfg_trainer["epochs"]
        self.early_stopping_count = self.cfg_trainer["early_stopping"]
        self.save_dir = self.cfg_trainer["save_dir"]
        self.learning_rate = self.cfg_trainer["learning_rate"]

        self.cfg_arch = self.config["arch"]
        self.model_name = self.cfg_arch["type"]
        self.n_users = self.cfg_arch["args"]["n_users"]
        self.n_items = self.cfg_arch["args"]["n_items"]

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.device = config["device"]  # main에서 추가

        self.optimizer = torch.optim.SparseAdam(
            params=self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = torch.nn.BCELoss()

        self.min_val_loss = float("inf")
        self.stopping_count = 0
        self.stopping = False

        self.gcs_helper = GCSHelper(
            "/opt/ml/final-project-level3-recsys-01/keys/gcs_key.json",
            "maple_trained_model",
        )

    def _train_epoch(self, epoch: int):
        self.model.train()

        total_train_loss = []
        print(f"...epoch {epoch} train...")
        for data in tqdm(self.train_data_loader):
            target = data["y"].to(self.device)
            output = self.model(data["x"]).cuda()

            loss = self.criterion(output.to(torch.float32), target.to(torch.float32))
            total_train_loss.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()

        total_val_loss = []
        for data in tqdm(self.valid_data_loader):
            target = data["y"].to(self.device)
            output = self.model(data["x"]).cuda()

            loss = self.criterion(output.to(torch.float32), target.to(torch.float32))
            total_val_loss.append(loss)

        train_loss = sum(total_train_loss) / len(total_train_loss)
        valid_loss = sum(total_val_loss) / len(total_val_loss)

        if self.min_val_loss > valid_loss:
            self.stopping_count = 0
            print(f"smaller valid loss... state has been updated")
            self.min_val_loss = valid_loss
            self.state = {
                "model_name": self.model_name,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
            }
        if self.stopping_count == self.early_stopping_count:
            self.stopping = True

        print(
            f"train_loss: {float(train_loss)}    valid_loss: {float(valid_loss.float())}    early stopping count: {self.stopping_count}/{self.early_stopping_count}"
        )

        self.stopping_count += 1

    def train(self):
        for epoch in range(self.epoch):
            self._train_epoch(epoch)
            if self.stopping:
                print("...early stopping...")
                break
        self._save_checkpoint()

    def _save_checkpoint(self):
        print(
            f"...SAVING MODEL...   model_name: {self.state['model_name']} epoch: {self.state['epoch']}"
        )
        save_path = os.path.join(self.save_dir, self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y%m%d-%H%M")
        save_path = os.path.join(save_path, f"{self.model_name}_{now}.pt")
        torch.save(self.state, save_path)
        if self.config["GCS_upload"]:
            self.gcs_helper.upload_model_to_gcs(
                f"{self.model_name}/{self.model_name}_{now}.pt", save_path
            )
