import torch
import torch.nn as nn
import os
from .auto_encoder import AutoEncoder
from src.utils.gcs_helper import GCSHelper


class AutoEncoderPredictor(nn.Module):
    def __init__(self, config, dropout_prop):
        super().__init__()
        self.config = config
        self.categories = ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]
        self.ae_models = dict()
        self.device = device
        self.get_models()
        self.dropout_prop = dropout_prop

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.predictor = nn.Sequential(
            nn.Linear(64 * len(self.categories), 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prop),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prop),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = list()

        for category, item in zip(self.categories, x):
            item_vector = self.ae_models[category](item)
            item_vector = self.avgpool(item_vector)
            item_vector = item_vector.view(item_vector.size(0), -1)

            output.append(item_vector)

        output = torch.cat(output, dim=-1)
        output = self.predictor(output)

        return output

    def get_models(self):
        gcs_helper = GCSHelper(
            key_path="src/utils/gcs_key.json", bucket_name="maple_trained_model"
        )
        for category in self.categories:
            auto_encoder = AutoEncoder()

            model_path = os.path.join(self.config["autoencoder_save_dir"], f"AutoEncoder_{category}.pt")

            if not os.path.exists(self.model_path):
                os.makedirs("/".join(self.model_path.split("/")[:-1]))
                gcs_helper.download_file_from_gcs(
                    blob_name=f"AutoEncoder/AutoEncoder_{category}.pt", file_name=self.model_path
                )

            auto_encoder.load_state_dict(torch.load(model_path))
            auto_encoder = auto_encoder.encoder

            for layer in auto_encoder.parameters():
                layer.requires_grad = False

            self.ae_models[category] = auto_encoder.to(self.device)