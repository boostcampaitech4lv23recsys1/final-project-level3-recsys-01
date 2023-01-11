import torch
import torch.nn as nn
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

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.device = config["device"] # main에서 추가

        self.optimizer = torch.optim.SparseAdam(params = self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.BCELoss()

    def _train_epoch(self, epoch:int):
        total_pred = []
        total_loss = []

        self.model.train()
        print(f"...epoch {epoch} train...")
        for data in tqdm(self.train_data_loader): # 547
            # breakpoint()
            # print(data)

            target = data["y"].to(self.device)
            output = self.model(data["x"])
            # interaction = data["y"]
            # items = torch.LongTensor(self.train_data[data])
            
            loss = self.criterion(output, target)
            total_pred.append(output)
            total_loss.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("pred: ", sum(total_pred)/self.n_users)
        print("loss: ", sum(total_loss)/self.n_users)

        total_pred = []
        total_loss = []
        self.model.eval()
        for data in tqdm(self.valid_data_loader):
            target = data["y"].to(self.device)
            output = self.model(data["x"], data["x"].groupby("user")['item'])

            loss = self.criterion(output, target)
            self.optimizer.zero_grad()

            total_pred.append(output)
            total_loss.append(loss)

        print("pred: ", sum(total_pred)/self.n_users)
        print("loss: ", sum(total_loss)/self.n_users)
    
    def train(self):
        for epoch in tqdm(range(self.epoch)):
            self._train_epoch(epoch)
        
        self.state = {
                    "model_name": self.model_name,
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                }

        self._save_checkpoint()
    
    def _save_checkpoint(self):
        print("...SAVING MODEL...")
        save_path = os.path.join(self.save_dir, "newMF.pt")
        torch.save(self.model.state_dict(), save_path)


