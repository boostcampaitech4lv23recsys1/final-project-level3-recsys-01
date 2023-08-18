from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

from modeling.model.resnet import get_resnet


class MCN(nn.Module):
    def __init__(
        self,
        embed_size=128,
        pe_off=False,
        pretrained=False,
        resnet_layer_num=18,
        hidden_sizes=[128],
        item_num=7,
    ):
        super(MCN, self).__init__()
        self.pe_off = pe_off
        self.mlp_layers = mlp_layers
        self.item_num = item_num

        print(f"using resnet{resnet_layer_num}...")
        self.cnn = get_resnet(layer_num=resnet_layer_num, pretrained=pretrained)
        self.cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        self.num_rela = 0
        for i in range(1, item_num + 1):
            self.num_rela += i
        self.num_rela *= 4
        self.bn = nn.BatchNorm1d(self.num_rela)

        predictor = []
        before_mlp_size = self.num_rela
        for hidden_size in range(hidden_sizes):
            linear = nn.Linear(before_mlp_size, hidden_size)
            nn.init.xavier_uniform_(linear.weight)
            predictor.append(linear)
            predictor.append(nn.ReLU())
            before_mlp_size = hidden_size

        linear = nn.Linear(before_mlp_size, 1)
        nn.init.xavier_uniform_(linear.weight)
        predictor.append(linear)
        self.predictor = nn.Sequential(*predictor)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(cnn.fc.weight)
        # nn.init.constant_(cnn.fc.bias, 0)

        self.masks = nn.Embedding(28, embed_size)
        self.masks.weight.data.normal_(0.9, 0.7)
        self.masks_l1 = nn.Embedding(28, 64)
        self.masks_l1.weight.data.normal_(0.9, 0.7)
        self.masks_l2 = nn.Embedding(28, 128)
        self.masks_l2.weight.data.normal_(0.9, 0.7)
        self.masks_l3 = nn.Embedding(28, 256)
        self.masks_l3.weight.data.normal_(0.9, 0.7)

        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images, names=None):
        out, features, tmasks, rep = self._compute_score(images)

        if self.pe_off:
            tmasks_loss, features_loss = torch.tensor(0.0), torch.tensor(0.0)
        else:
            tmasks_loss, features_loss = self._compute_type_repr_loss(tmasks, features)

        return out, tmasks_loss, features_loss

    def _compute_type_repr_loss(self, tmasks, features):
        # Type embedding loss
        tmasks_loss = tmasks.norm(1) / len(tmasks)
        features_loss = features.norm(2) / np.sqrt((features.shape[0] * features.shape[1]))

        return tmasks_loss, features_loss

    def _compute_score(self, images, activate=True):
        batch_size, item_num, channels, h, w = images.shape
        images = torch.reshape(images, (-1, channels, h, w))

        features, rep_l1, rep_l2, rep_l3, rep_l4, rep  = self.cnn(images)

        relations = []
        features = features.reshape(batch_size, item_num, -1)
        masks = F.relu(self.masks.weight)

        for mi, (item_i, item_j) in enumerate(combinations_with_replacement([i for i in range(self.item_num)], 2)):
            if self.pe_off:
                left = F.normalize(features[:, item_i, :], dim=-1)
                right = F.normalize(features[:, item_j, :], dim=-1)
            else:
                left = F.normalize(masks[mi] * features[:, item_i:item_i + 1, :], dim=-1)
                right = F.normalize(masks[mi] * features[:, item_j:item_j + 1, :], dim=-1)
            rela = torch.matmul(left, right.transpose(1, 2)).squeeze()
            relations.append(rela)

        rep_list, masks_list = list(), list()

        rep_list.append(rep_l1)
        masks_list.append(self.masks_l1)
        rep_list.append(rep_l2)
        masks_list.append(self.masks_l2)
        rep_list.append(rep_l3)
        masks_list.append(self.masks_l3)

        for rep_li, masks_li in zip(rep_list, masks_list):
            rep_li = (self.ada_avgpool2d(rep_li).squeeze().reshape(batch_size, item_num, -1))

            for mi, (i, j) in enumerate(combinations_with_replacement([i for i in range(self.item_num)], 2)):
                left = F.normalize(masks_li[mi] * rep_li[:, i, :], dim=-1)
                right = F.normalize(masks_li[mi] * rep_li[:, j, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze()
                relations.append(rela)

        if batch_size == 1:
            relations = torch.stack(relations).unsqueeze(0)
        else:
            relations = torch.stack(relations, dim=1)
        relations = self.bn(relations)

        out = self.predictor(relations)
        out = self.sigmoid(out)

        return out, features, masks, rep
