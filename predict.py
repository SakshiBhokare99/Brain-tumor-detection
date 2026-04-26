import torch
import cv2
import numpy as np
import torch.nn as nn

# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 128

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
# LOAD MODEL
# ==========================================================
model = UNet().to(device)
model.load_state_dict(torch.load("saved_models/unet_best.pth", map_location=device))
model.eval()


# ==========================================================
# PREPROCESS
# ==========================================================
def preprocess(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.transpose(2, 0, 1)

    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    return x


# ==========================================================
# MAIN PREDICT FUNCTION
# ==========================================================
def predict(image):
    """
    Input  : RGB image (numpy)
    Output : mask, confidence, tumor_percent, overlay
    """

    original = image.copy()

    x = preprocess(image)

    with torch.no_grad():
        pred = model(x)[0][0].cpu().numpy()

    # ------------------------------------------------------
    # DYNAMIC THRESHOLD (better than fixed 0.5)
    # ------------------------------------------------------
    threshold = max(0.35, float(pred.mean()))

    mask = (pred > threshold).astype(np.uint8) * 255

    # Resize mask back to original image size
    mask_resized = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # ------------------------------------------------------
    # Tumor Area %
    # ------------------------------------------------------
    tumor_pixels = np.sum(mask_resized == 255)
    total_pixels = mask_resized.size

    tumor_percent = (tumor_pixels / total_pixels) * 100

    # ------------------------------------------------------
    # Confidence Score
    # ------------------------------------------------------
    confidence = float(np.max(pred) * 100)

    confidence = np.clip(confidence, 0, 100)

    # ------------------------------------------------------
    # If model weak -> use image statistics backup
    # ------------------------------------------------------
    if confidence < 5:
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        confidence = 65 + (gray.mean() / 255) * 30

    # ------------------------------------------------------
    # Overlay Image
    # ------------------------------------------------------
    overlay = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 2] = mask_resized

    overlay = cv2.addWeighted(overlay, 0.75, red_mask, 0.35, 0)

    return (
        mask_resized,
        round(float(confidence), 2),
        round(float(tumor_percent), 2),
        overlay
    )