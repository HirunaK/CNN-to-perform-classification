import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ModelB(nn.Module):
    def __init__(self, num_classes=5):
        super(ModelB, self).__init__()
        self.features = nn.Sequential(
            # Initial light standard conv or depthwise
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 32 x 32

            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2, 2),  # 32 x 16 x 16

            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2)   # 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

if __name__ == '__main__':
    model = ModelB()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model B Total Trainable Parameters: {total_params}")
    assert total_params < 10000000, "Model B exceeds the 100,000 parameter budget!"