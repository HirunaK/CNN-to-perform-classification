import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.model_a import ModelA


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 64


class FlowerDataset(Dataset):
    def __init__(self, split_dir, class_to_idx):
        self.class_to_idx = class_to_idx
        self.samples = [
            (image_path, class_to_idx[class_name])
            for class_name in class_to_idx
            for image_path in sorted((split_dir / class_name).iterdir())
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        return image, label


def find_data_dir(project_dir):
    candidates = [project_dir / "data" / "data", project_dir / "data"]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find train/ and val/ folders under data/ or data/data/. "
        "Run data/dataset_loader.py first."
    )


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_function(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def main():
    parser = argparse.ArgumentParser(description="Train Model A on the flower dataset.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model_a.pth"))
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or find_data_dir(project_dir)
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    val_classes = {path.name for path in val_dir.iterdir() if path.is_dir()}
    if set(class_names) != val_classes:
        raise ValueError("Train and validation folders must contain the same classes")

    train_dataset = FlowerDataset(train_dir, class_to_idx)
    val_dataset = FlowerDataset(val_dir, class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelA(num_classes=len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    output_path = args.output if args.output.is_absolute() else project_dir / args.output

    print(f"Training on {device} with {len(train_dataset)} train and {len(val_dataset)} validation images")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_function, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": IMAGE_SIZE,
                },
                output_path,
            )

    print(f"Best model saved to {output_path}")


if __name__ == "__main__":
    main()
