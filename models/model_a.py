import torch
import torch.nn as nn

class ModelA(nn.Module):
    def __init__(self, num_classes=5):
        super(ModelA, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: 3 x 64 x 64 -> 32 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 32 x 32

            # Layer 2: 32 x 32 x 32 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 16 x 16

            # Layer 3: 64 x 16 x 16 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 128 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model_a = ModelA()
    print(f"Model A Trainable Parameters: {count_parameters(model_a):,}")