import torch
import os
from tqdm import tqdm


class newMFTrainer:
    def __init__(self, config, model, train_data_loader, valid_data_loader):
        self.model = model

        self.config = config
        self.cfg_trainer = config["trainer"]
        self.epoch = self.cfg_trainer["epochs"]
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

        print("train_loss: ", train_loss, "valid_loss: ", valid_loss)
        if self.min_val_loss > valid_loss:
            print(f"smaller valid loss... state has been updated")
            self.min_val_loss = valid_loss
            self.state = {
                "model_name": self.model_name,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
            }

    def train(self):
        for epoch in range(self.epoch):
            self._train_epoch(epoch)
        self._save_checkpoint()

    def _save_checkpoint(self):
        print("...SAVING MODEL...")
        save_path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        torch.save(self.state, save_path)
