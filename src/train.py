import argparse, json, os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms as T
from PIL import Image
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # Filter out corrupt images
        valid_samples = []
        print(f"\nVerifying images in {root}")
        for path, target in tqdm(self.samples, desc="Checking images"):
            try:
                with open(path, "rb") as f:
                    Image.open(f).verify()
                valid_samples.append((path, target))
            except Exception as e:
                print(f"\nWarning: Skipping corrupted image {path}: {str(e)}")
        self.samples = valid_samples
        self.imgs = valid_samples
        print(f"Found {len(valid_samples)} valid images")


def make_datasets(data_dir: str, val_ratio: float = 0.2):
    train_tf = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    if val_dir.exists():
        train_ds = SafeImageFolder(train_dir, transform=train_tf)
        val_ds = SafeImageFolder(val_dir, transform=eval_tf)
        classes = train_ds.classes
    else:
        full = SafeImageFolder(train_dir, transform=train_tf)
        classes = full.classes
        n_val = int(len(full) * val_ratio)
        n_train = len(full) - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val])
        # Override val transform
        val_ds.dataset.transform = eval_tf
    return train_ds, val_ds, classes


def build_model(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    progress = tqdm(loader, desc="Training")
    for x, y in progress:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        # Update progress bar with current loss
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    return running / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct = 0.0, 0
    progress = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            # Update progress bar
            current_acc = correct / ((progress.n + 1) * loader.batch_size)
            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"}
            )
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir", required=True, help="Path with train/ and optional val/"
    )
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, classes = make_datasets(args.data_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_acc > best_acc:
            best_acc, best_state = va_acc, model.state_dict()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(best_state or model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "class_indices.json"), "w") as f:
        json.dump(list(classes), f)
    print(f"Saved model + classes to {args.out_dir}")


if __name__ == "__main__":
    main()
