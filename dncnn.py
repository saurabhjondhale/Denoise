import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):
        super().__init__()
        layers = [nn.Conv2d(channels, 64, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(num_layers-2):
            layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1),
                       nn.BatchNorm2d(64),
                       nn.ReLU()]
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  # Residual learning