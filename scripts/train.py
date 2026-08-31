"""Train FANet with feedback attention.

Usage:
    python scripts/train.py [--config configs/kvasir_sessile.yaml]
"""
import argparse
import datetime
import os
import time

import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader

from fanet.config import load_config
from fanet.data import DATASET, load_data, rle_batch_to_tensor
from fanet.losses import DiceBCELoss
from fanet.models import FANet
from fanet.utils import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, print_and_save,
)


def train(model, loader, mask, optimizer, loss_fn, device, size, dual_path=False):
    epoch_loss = 0
    return_mask = []

    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        b = y.shape[0]
        m = rle_batch_to_tensor(mask, i * b, b, size)
        if dual_path:
            bg = (1.0 - m)
            m = torch.cat([m, bg], dim=1)
        m = m.to(device)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            for py in y_pred:
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                return_mask.append(rle_encode(py))

        epoch_loss += loss.item()

    return epoch_loss / len(loader), return_mask


def evaluate(model, loader, mask, loss_fn, device, size, dual_path=False):
    epoch_loss = 0
    return_mask = []

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            b = y.shape[0]
            m = rle_batch_to_tensor(mask, i * b, b, size)
            if dual_path:
                bg = (1.0 - m)
                m = torch.cat([m, bg], dim=1)
            m = m.to(device)

            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            for py in y_pred:
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                return_mask.append(rle_encode(py))

    return epoch_loss / len(loader), return_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint path to resume from")
    parser.add_argument("--gate", type=str, default="binary",
                        choices=["binary", "ste", "soft"],
                        help="MixPool gate mode (Phase 2/4)")
    parser.add_argument("--dual-path", action="store_true",
                        help="dual-path feedback [m_fg, m_bg] (Phase 3/4)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override num epochs (smoke test)")
    parser.add_argument("--loss", type=str, default="dicebce",
                        choices=["dicebce", "negdice"],
                        help="loss fn (fallback F1: negdice = negative-area Dice)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["train"]["batch_size"]
    num_epochs = args.epochs or cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    checkpoint_path = cfg["paths"]["checkpoint"]
    train_log_path = cfg["paths"]["train_log"]
    aug = cfg["train"]["augmentation"]
    gate = args.gate
    dual_path = args.dual_path

    if gate != "binary" or dual_path or args.loss != "dicebce":
        base, ext = os.path.splitext(checkpoint_path)
        checkpoint_path = f"{base}_{gate}_{'dual' if dual_path else 'single'}_{args.loss}{ext}"
        train_log_path = (f"{os.path.splitext(train_log_path)[0]}_{gate}_"
                          f"{'dual' if dual_path else 'single'}_{args.loss}.txt")

    seeding(cfg["train"]["seed"])

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)

    print_and_save(train_log_path, str(datetime.datetime.now()))

    (train_x, train_y), (valid_x, valid_y) = load_data(cfg["dataset"]["root"])
    train_x, train_y = shuffling(train_x, train_y)
    print_and_save(train_log_path,
                   f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n")

    transform = A.Compose([
        A.Rotate(limit=aug["rotate_limit"], p=aug["rotate_p"]),
        A.HorizontalFlip(p=aug["hflip_p"]),
        A.VerticalFlip(p=aug["vflip_p"]),
        A.CoarseDropout(p=aug["dropout_p"], num_holes=10,
                        hole_height=32, hole_width=32),
    ])

    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FANet(gate=gate, dual_path=dual_path).to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print_and_save(train_log_path, f"Resumed from {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    if args.loss == "negdice":
        from fanet.losses import NegativeAreaDiceBCELoss
        loss_fn = NegativeAreaDiceBCELoss(fp_weight=0.5)
        loss_name = "NegativeAreaDiceBCE (fallback F1)"
    else:
        loss_fn = DiceBCELoss()
        loss_name = "BCE Dice Loss"

    print_and_save(train_log_path,
                   f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}"
                   f"\nLR: {lr}\nEpochs: {num_epochs}\nOptimizer: Adam\nLoss: {loss_name}\n")

    best_valid_loss = float('inf')
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, return_train_mask = train(
            model, train_loader, train_mask, optimizer, loss_fn, device, size, dual_path)
        valid_loss, return_valid_mask = evaluate(
            model, valid_loader, valid_mask, loss_fn, device, size, dual_path)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print_and_save(train_log_path, f"Saving checkpoint: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            train_mask = return_train_mask
            valid_mask = return_valid_mask

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print_and_save(train_log_path, data_str)


if __name__ == "__main__":
    main()
