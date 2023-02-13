import torch
from torch import nn

from typing import Tuple


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 3 by 3 이랑 1 by 1 한번씩 통과시켜서 channel 수 늘리기
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 일단 패스
        # out += residual
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(self, pretrain=False) -> None:
        super().__init__()

        # pretrain 과정에서는 1~7의 카테고리를 내뱉도록 변경
        self.pretrain = pretrain

        # (24, 24, 3) -> (24, 24, 32)
        self.layer1 = nn.Sequential(
            BasicBlock(3, 32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2),
            nn.BatchNorm2d(64),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool2d((2, 2))
        self.avgpool2 = nn.AdaptiveAvgPool2d((2, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_pretrain = nn.Linear(128, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        batch_size, channels_size, h, w = x.shape

        # (24, 24, 3) -> (12, 12, 32)
        x = self.layer1(x)
        rep_1 = self.avgpool1.forward(x)  # (2, 2, 32)
        rep_1 = rep_1.squeeze().view(batch_size, -1)  # (batch, 128)

        # (12, 12, 32) -> (6, 6, 64)
        x = self.layer2(x)
        rep_2 = self.avgpool2.forward(x)  # (2, 1, 64)
        rep_2 = rep_2.squeeze().view(batch_size, -1)  # (batch, 128)

        # (6, 6, 64) -> (3, 3, 128)
        x = self.layer3(x)
        rep_3 = self.avgpool3.forward(x)  # (1, 1, 128)
        rep_3 = rep_3.squeeze().view(batch_size, -1)  # (batch, 128)

        if self.pretrain:
            # (3, 3, 128) -> (1, 1, 128)
            x = self.maxpool.forward(x)
            x = x.squeeze().view(batch_size, -1)
            return self.fc_pretrain(x)  # (batch, 7)

        return rep_1, rep_2, rep_3


class SimpleFullyConnected(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x


class SimpleMCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = SimpleCNN()
        self.fc = SimpleFullyConnected(128 * 3, 128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x: (batch, 7, channels, h, w)
        batch_size, item_num, channels, h, w = x.shape

        # 7개 사진 배치처럼 묶어서 처리
        x = x.view(-1, channels, h, w)
        rep_1, rep_2, rep_3 = self.cnn.forward(x)

        # 나온 결과는 다시 7개 사진으로 재분배
        rep_1 = rep_1.view(batch_size, item_num, -1)
        rep_2 = rep_2.view(batch_size, item_num, -1)
        rep_3 = rep_3.view(batch_size, item_num, -1)

        # 각 layer 결과 그냥 평균내버리기 (batch, 7, -1) -> (batch, -1)
        rep_1 = rep_1.mean(dim=1)
        rep_2 = rep_2.mean(dim=1)
        rep_3 = rep_3.mean(dim=1)

        # 평균 낸 layer 합쳐서 Fully connected layer 통과시켜 결과 내기
        # (batch, x) -> (batch, 3x)
        concat = torch.cat((rep_1, rep_2, rep_3), dim=1)
        output = self.fc.forward(concat).squeeze()

        return self.sigmoid(output)
