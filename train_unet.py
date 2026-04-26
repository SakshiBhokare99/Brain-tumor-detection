import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# SETTINGS
# ==========================================================
IMG_SIZE = 96
BATCH_SIZE = 8
EPOCHS = 8
MAX_IMAGES = 1500

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

if device == "cuda":
    torch.backends.cudnn.benchmark = True


# ==========================================================
# DATASET
# ==========================================================
class MRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:MAX_IMAGES]

        print("Images Loaded:", len(self.images))

        self.img_paths = [os.path.join(img_dir, f) for f in self.images]
        self.mask_paths = [os.path.join(mask_dir, f) for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)

        # ⚠️ safety check (prevents worker crash)
        if img is None or mask is None:
            return self.__getitem__((idx + 1) % len(self))

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # normalize
        img = img.transpose(2, 0, 1) / 255.0
        mask = np.expand_dims(mask, 0) / 255.0

        img = (img - 0.5) / 0.5

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )


# ==========================================================
# MODEL
# ==========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.u2 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))

        x = self.up1(c3)
        x = torch.cat([x, c2], dim=1)
        x = self.u1(x)

        x = self.up2(x)
        x = torch.cat([x, c1], dim=1)
        x = self.u2(x)

        return torch.sigmoid(self.out(x))


# ==========================================================
# LOSS
# ==========================================================
bce = nn.BCELoss()

def dice_loss(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)

    pred = torch.clamp(pred, 0, 1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)


# ==========================================================
# DATA LOADING
# ==========================================================
def get_loader():
    dataset = MRIDataset("dataset/images", "dataset/masks")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # WINDOWS SAFE
        pin_memory=(device == "cuda"),
        persistent_workers=True
    )
    return loader


# ==========================================================
# TRAINING FUNCTION
# ==========================================================
def train():
    loader = get_loader()

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Started...\n")

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = combined_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch+1}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/unet_best.pth")
            print("Best Model Saved!\n")

    print("Training Completed!")


# ==========================================================
# MAIN ENTRY (IMPORTANT FOR WINDOWS)
# ==========================================================
if __name__ == "__main__":
    train()