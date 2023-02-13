import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(
            self,
            stride=2,
            padding=1,
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=stride, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=stride, padding=padding, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=stride, padding=padding, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def reset_parameters(self):
        def weight_reset(layer):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                layer.reset_parameters()

        self.encoder.apply(weight_reset)
        self.decoder.apply(weight_reset)
