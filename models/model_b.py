import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise Conv (spatial filtering)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=1, groups=in_channels, bias=False
        )
        # Pointwise Conv (channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
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
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16 x 32 x 32

            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2, 2),  # -> 32 x 16 x 16

            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2),  # -> 64 x 8 x 8

            DepthwiseSeparableConv(64, 64),
            nn.MaxPool2d(2, 2)   # -> 64 x 4 x 4 (Extra maxpool reduces flatten size to 1024)
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 64 * 4 * 4 = 1024
            nn.Linear(1024, 32),              # 1024 * 32 + 32 = 32,800 params
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)        # 32 * 5 + 5 = 165 params
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = ModelB()
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters in Model B: {total_params:,}")
    assert total_params < 100000, "Model B exceeds parameter limit!"