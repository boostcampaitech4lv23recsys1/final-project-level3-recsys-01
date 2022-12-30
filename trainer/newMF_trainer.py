import torch
import torch.nn as nn
import os
from tqdm import tqdm

class newMFTrainer:
    def __init__(self, model, train_data, config):
        self.model = model
        self.config = config
        self.cfg_trainer = config["trainer"]
        self.epoch = self.cfg_trainer["epochs"]
        self.save_dir = self.cfg_trainer["save_dir"]
        self.train_data = train_data
        self.n_users = len(self.train_data)
        print("bbb")
        self.optimizer = torch.optim.SparseAdam(self.model.parameters(self), lr=0.001)
        self.criterion = torch.nn.BCELoss()

    def __train_epoch(self, epoch:int):
        total_pred = []
        total_loss = []
        print(f"...epoch {epoch} train...")
        for user in range(self.n_users):
            self.optimizer.zero_grad()

            interaction = torch.Tensor([1])
            items = torch.LongTensor(self.train_data[user])
            
            prediction = self.model(user, items)
            loss = self.criterion(prediction, interaction[0])
            total_pred.append(prediction)
            total_loss.append(loss)

            loss.backward()

            self.optimizer.step()
        print("pred: ", sum(total_pred)/self.n_users)
        print("loss: ", sum(total_loss)/self.n_users)
    
    def train(self):
        for epoch in tqdm(range(self.epoch)):
            self.__train_epoch(epoch)
        
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


