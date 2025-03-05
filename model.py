from torch import nn
import torch
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.layers=nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.layers(x)

class ConvolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(
            ConvolutionBlock(3, 32),  # 32x32 → 16x16
            ConvolutionBlock(32, 64),  # 16x16 → 8x8
            ConvolutionBlock(64, 128),  # 8x8 → 4x4
            ConvolutionBlock(128, 256),  # 4x4 → 2x2
            ConvolutionBlock(256, 512),  # 2x2 → 1x1
        )
        self.layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
            #nn.Softmax(dim=1),
        )
    def forward(self, x):
        z=self.block(x)
        y=self.layers(z)
        return y
