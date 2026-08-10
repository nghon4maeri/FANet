"""Generate FANet report data with real Kvasir-SEG dataset."""
import os, time, numpy as np, torch, cv2
from torch.utils.data import DataLoader
from sklearn.utils import shuffle as sk_shuffle
import albumentations as A

from utils import (
    seeding, create_dir, init_mask, epoch_time,
    rle_encode, rle_decode, print_and_save, load_data
)
from model import FANet
from loss import DiceBCELoss

IMG_SIZE = (256, 256)
BATCH = 2
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Dataset Class ==========
class DATASET(torch.utils.data.Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        image = cv2.resize(image, IMG_SIZE)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = image.astype(np.float32)
        mask = cv2.resize(mask, IMG_SIZE)
        mask = np.expand_dims(mask, axis=0) / 255.0
        mask = mask.astype(np.float32)
        return image, mask

    def __len__(self):
        return self.n_samples

# ========== Training Functions ==========
def train_epoch(model, loader, mask, optimizer, loss_fn, device):
    epoch_loss = 0
    return_mask = []
    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        b = y.shape[0]
        m = []
        for edata in mask[i*BATCH : i*BATCH + b]:
            edata = " ".join(str(d) for d in edata)
            edata = str(edata)
            edata = rle_decode(edata, IMG_SIZE)
            edata = np.expand_dims(edata, axis=0)
            m.append(edata)
        m = np.array(m, dtype=np.int32)
        m = np.transpose(m, (0, 1, 3, 2))
        m = torch.from_numpy(m).to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_bin = (torch.sigmoid(y_pred) > 0.5).cpu().numpy().astype(np.uint8)
            for py in y_pred_bin:
                return_mask.append(rle_encode(np.squeeze(py, axis=0)))
        epoch_loss += loss.item()
    return epoch_loss / len(loader), return_mask

def eval_epoch(model, loader, mask, loss_fn, device):
    epoch_loss = 0
    return_mask = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            b = y.shape[0]
            m = []
            for edata in mask[i*BATCH : i*BATCH + b]:
                edata = " ".join(str(d) for d in edata)
                edata = str(edata)
                edata = rle_decode(edata, IMG_SIZE)
                edata = np.expand_dims(edata, axis=0)
                m.append(edata)
            m = np.array(m, dtype=np.int32)
            m = np.transpose(m, (0, 1, 3, 2))
            m = torch.from_numpy(m).to(device, dtype=torch.float32)

            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred_bin = (torch.sigmoid(y_pred) > 0.5).cpu().numpy().astype(np.uint8)
            for py in y_pred_bin:
                return_mask.append(rle_encode(np.squeeze(py, axis=0)))
    return epoch_loss / len(loader), return_mask

# ========== MAIN ==========
if __name__ == "__main__":
    seeding(42)
    create_dir("files")
    log_path = "files/report_log.txt"
    print_and_save(log_path, f"Report generated: {time.ctime()}")

    # Config
    print_and_save(log_path, "\n===== CONFIGURATION =====")
    print_and_save(log_path, f"Device:       {DEVICE}")
    print_and_save(log_path, f"Image size:   {IMG_SIZE}")
    print_and_save(log_path, f"Batch size:   {BATCH}")
    print_and_save(log_path, f"Epochs:       {EPOCHS}")
    print_and_save(log_path, f"Learning rate:{LR}")
    print_and_save(log_path, f"Optimizer:    Adam")
    print_and_save(log_path, f"Loss:         DiceBCELoss (0.5*Dice + 0.5*BCE)")

    # Model
    model = FANet().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print_and_save(log_path, f"\n===== MODEL ARCHITECTURE =====")
    print_and_save(log_path, f"Architecture: U-Net Encoder-Decoder (4 stages)")
    print_and_save(log_path, f"Encoder: 3 -> 32 -> 64 -> 128 -> 256 channels, MaxPool 2x2")
    print_and_save(log_path, f"Decoder: 256 -> 128 -> 64 -> 32 -> 16 channels, ConvTranspose2d")
    print_and_save(log_path, f"Output: Conv1x1 (17 -> 1) + Sigmoid")
    print_and_save(log_path, f"Total parameters: {total_params:,}")

    # Per-block param count
    blocks = [
        ("Encoder 1 (3->32)", model.e1),
        ("Encoder 2 (32->64)", model.e2),
        ("Encoder 3 (64->128)", model.e3),
        ("Encoder 4 (128->256)", model.e4),
        ("Decoder 1 (256->128)", model.d1),
        ("Decoder 2 (128->64)", model.d2),
        ("Decoder 3 (64->32)", model.d3),
        ("Decoder 4 (32->16)", model.d4),
        ("Output (Conv1x1)", model.output),
    ]
    print_and_save(log_path, f"\nParameter breakdown:")
    for name, blk in blocks:
        n = sum(p.numel() for p in blk.parameters())
        print_and_save(log_path, f"  {name:<25s}: {n:>10,}")

    # Dataset
    path = "./sessile-main-Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    train_x, train_y = sk_shuffle(train_x, train_y, random_state=42)

    print_and_save(log_path, f"\n===== DATASET =====")
    print_and_save(log_path, f"Dataset:     Kvasir-SEG (polyp segmentation)")
    print_and_save(log_path, f"Train:       {len(train_x)} images")
    print_and_save(log_path, f"Validation:  {len(valid_x)} images")
    print_and_save(log_path, f"Image format:JPEG, 256x256 resized")

    # Show sample filename
    sample_img = train_x[0].split("/")[-1]
    print_and_save(log_path, f"Sample ID:   {sample_img}")

    # DataLoader
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
    ])
    train_dataset = DATASET(train_x, train_y, IMG_SIZE, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, IMG_SIZE, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()

    # Init masks
    print_and_save(log_path, f"\n===== INITIAL MASKS (OTSU) =====")
    print_and_save(log_path, f"Generating initial masks via Otsu thresholding...")
    train_mask = init_mask(train_x, IMG_SIZE)
    valid_mask = init_mask(valid_x, IMG_SIZE)
    print_and_save(log_path, f"Done. RLE-encoded masks ready.")

    # Training
    print_and_save(log_path, f"\n===== TRAINING RESULTS ({EPOCHS} epochs) =====")
    print_and_save(log_path, f"{'Ep':<5}{'Train Loss':<14}{'Val Loss':<14}{'Best Val':<14}{'Time(s)':<10}{'LR':<10}{'Note'}")
    print_and_save(log_path, f"{'-'*70}")

    best_valid_loss = float('inf')
    ckpt_path = "files/checkpoint.pth"

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, return_train_mask = train_epoch(model, train_loader, train_mask, optimizer, loss_fn, DEVICE)
        valid_loss, return_valid_mask = eval_epoch(model, valid_loader, valid_mask, loss_fn, DEVICE)
        scheduler.step(valid_loss)

        improved = ""
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)
            train_mask = return_train_mask
            valid_mask = return_valid_mask
            improved = " * (mask updated)"

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print_and_save(log_path,
            f"{epoch+1:<5}{train_loss:<14.4f}{valid_loss:<14.4f}{best_valid_loss:<14.4f}{elapsed:<10.1f}{lr_now:<10.1e}{improved}")

    print_and_save(log_path, f"\nBest validation loss: {best_valid_loss:.4f}")
    print_and_save(log_path, f"Checkpoint saved:     {ckpt_path}")

    # Test-time refinement demo
    print_and_save(log_path, f"\n===== TEST-TIME REFINEMENT (1 sample) =====")
    model.eval()

    test_img_path = valid_x[0]
    img_name = test_img_path.split("/")[-1]
    gt_mask_path = valid_y[0]

    img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(DEVICE)

    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, IMG_SIZE) / 255.0
    gt_fg = (gt > 0.5).sum()

    # Otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pm = (th / 255.0 > 0.5).astype(np.float32)
    pt = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    print_and_save(log_path, f"Test image:            {img_name}")
    print_and_save(log_path, f"Ground truth FG pixels: {gt_fg:.0f}")
    print_and_save(log_path, f"")
    print_and_save(log_path, f"{'Iter':<8}{'FG Pixels':<15}{'vs GT'}")
    print_and_save(log_path, f"{'-'*35}")
    print_and_save(log_path, f"{'Otsu':<8}{pm.sum():<15.0f}{pm.sum()-gt_fg:+.0f}")

    for it in range(5):
        with torch.no_grad():
            pred = torch.sigmoid(model([img_tensor, pt]))
            pred_bin = (pred > 0.5).float()
            n_fg = pred_bin.sum().item()
            print_and_save(log_path, f"{it+1:<8}{n_fg:<15.0f}{n_fg-gt_fg:+.0f}")
            pt = pred_bin

    print_and_save(log_path, f"\nReport complete.")
