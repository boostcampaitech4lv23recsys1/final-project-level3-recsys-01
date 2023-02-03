import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datetime import datetime

from pytz import timezone
from sklearn import metrics
from modeling.trainer.loss import get_loss
from modeling.trainer.scheduler import get_scheduler
from modeling.utils.gcs_helper import GCSHelper

import wandb


class MCNTrainer(object):
    def __init__(
        self,
            config,
            model,
            train_loader,
            val_loader,
    ):
        self.config = config
        self.model = model
        self.device = config["device"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = get_loss(config["trainer"])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9)
        self.scheduler = get_scheduler(self.optimizer, config["trainer"])
        self.epochs = config["trainer"]["epochs"]
        self.best = float("-inf")
        self.print_every = config["trainer"]["print_every"]

        self.model_name = "MCN"
        self.gcs_helper = GCSHelper(
            key_path="keys/gcs_key.json",
            bucket_name="maple_trained_model",
        )

    def train(self):
        self.model = self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            print("Train Phase, Epoch: {}".format(epoch))
            self.scheduler.step()

            total_losses = AverageMeter()
            clf_losses = AverageMeter()
            vse_losses = AverageMeter()

            self.model.train()
            accuracy = self.__train(
                epoch=epoch,
                total_losses=total_losses,
                clf_losses=clf_losses,
                vse_losses=vse_losses,
            )

            if accuracy > self.best:
                self.best = accuracy

                now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y%m%d-%H%M")
                model_save_path = (
                    self.config["trainer"]["save_dir"] + f"/{self.model_name}_{now}.pt"
                )
                torch.save(self.model.state_dict(), model_save_path)
                print("Saved best model to {}".format(model_save_path))

                if self.config["GCS_upload"]:
                    self.gcs_helper.upload_model_to_gcs(
                        f"{self.model_name}/{self.model_name}_{now}.pt", model_save_path
                    )

    def __train(self, epoch, total_losses, clf_losses, vse_losses):
        for batch_num, batch in enumerate(self.train_loader, 1):
            images, is_compat = batch
            images = images.to(self.device)

            output, vse_loss, tmasks_loss, features_loss = self.model(images)

            target = is_compat.float().to(self.device)
            output = output.squeeze(dim=1)
            clf_loss = self.criterion(output, target)

            features_loss = 5e-3 * features_loss
            tmasks_loss = 5e-4 * tmasks_loss
            total_loss = clf_loss + vse_loss + features_loss + tmasks_loss

            total_losses.update(total_loss.item(), images.shape[0])
            clf_losses.update(clf_loss.item(), images.shape[0])
            vse_losses.update(vse_loss.item(), images.shape[0])

            self.model.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if batch_num % self.print_every == 0:
                print(
                    "[{}/{}] #{} clf_loss: {:.4f}, vse_loss: {:.4f}, features_loss: {:.4f}, tmasks_loss: {:.4f}, total_loss:{:.4f}".format(
                        epoch,
                        self.epochs,
                        batch_num,
                        clf_losses.val,
                        vse_losses.val,
                        features_loss,
                        tmasks_loss,
                        total_losses.val,
                    )
                )
                log = {
                    "clf_loss": clf_losses.val,
                    "vse_loss": vse_losses.val,
                    "features_loss": features_loss,
                    "tmasks_loss": tmasks_loss,
                    "total_loss": total_losses.val,
                }
                wandb.log(log, step=batch_num)
        print("Train Loss (clf_loss): {:.4f}".format(clf_losses.avg))

        epoch_train_log = {
            "train_clf_loss": clf_losses.avg,
            "train_vse_loss": vse_losses.avg,
            "train_total_loss": total_losses.avg,
        }
        wandb.log(epoch_train_log, step=epoch)

        accuracy = self.__val(epoch)


        return accuracy

    def __val(self, epoch):
        print("Valid Phase, Epoch: {}".format(epoch))
        self.model.eval()

        clf_losses = AverageMeter()
        outputs = []
        targets = []
        for batch_num, batch in enumerate(self.val_loader, 1):
            images, is_compat = batch
            images = images.to(self.device)
            target = is_compat.float().to(self.device)
            with torch.no_grad():
                output, _, _, _ = self.model._compute_score(images)
                output = output.squeeze(dim=1)
                clf_loss = self.criterion(output, target)
            clf_losses.update(clf_loss.item(), images.shape[0])
            outputs.append(output)
            targets.append(target)
        print("Valid Loss (clf_loss): {:.4f}".format(clf_losses.avg))
        outputs = torch.cat(outputs).cpu().data.numpy()
        targets = torch.cat(targets).cpu().data.numpy()
        predicts = np.where(outputs > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(predicts, targets)
        print("Accuracy@0.5: {:.4f}".format(accuracy))
        positive_loss = -np.log(outputs[targets == 1]).mean()
        print("Positive loss: {:.4f}".format(positive_loss))
        positive_acc = sum(outputs[targets == 1] > 0.5) / len(outputs)
        print("Positive accuracy: {:.4f}".format(positive_acc))

        epoch_valid_log = {
            "valid_clf_loss": clf_losses.avg,
            "accuracy@0.5": accuracy,
            "positive_loss": positive_loss,
            "positive_accuracy": positive_acc,
        }
        wandb.log(epoch_valid_log, step=epoch)

        return accuracy


class AverageMeter(object):
    """Computes and stores the average and current value.
    >>> acc = AverageMeter()
    >>> acc.update(0.6)
    >>> acc.update(0.8)
    >>> print(acc.avg)
    0.7
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
