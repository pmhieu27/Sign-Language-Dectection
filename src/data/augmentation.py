"""
Step 4: Data Augmentation — Chỉ áp dụng trên Train set
Lưu ra file RIÊNG (X_train_aug.npy), KHÔNG ghi đè file gốc.

Usage: python src/augmentation.py
"""

import numpy as np
import os
from scipy.interpolate import interp1d

from src.config import PROCESSED_PATH, COORD_FEATURES, RAW_FEATURES, NUM_FEATURES
from src.data.preprocess import compute_inter_hand_features


def _split_processed_sequence(seq):
    coords = seq[:, :COORD_FEATURES].copy()
    presence = seq[:, COORD_FEATURES:RAW_FEATURES].copy()
    return coords, presence


def _rebuild_processed_sequence(coords, presence):
    left_block = coords[:, :63]
    right_block = coords[:, 63:126]
    inter = compute_inter_hand_features(left_block, right_block, presence)
    rebuilt = np.concatenate([coords, presence, inter], axis=1).astype(np.float32)
    if rebuilt.shape[1] != NUM_FEATURES:
        raise ValueError(f"Unexpected feature size: {rebuilt.shape[1]} != {NUM_FEATURES}")
    return rebuilt


def speed_variation(seq, factor):
    """Thay đổi tốc độ — resample sequence với hệ số factor."""
    old_len = len(seq)
    new_len = max(2, int(old_len * factor))
    f = interp1d(np.arange(old_len), seq, axis=0)
    resampled = f(np.linspace(0, old_len - 1, new_len))
    f2 = interp1d(np.arange(new_len), resampled, axis=0)
    return f2(np.linspace(0, new_len - 1, old_len)).astype(np.float32)


def gaussian_jitter(seq, sigma=0.01):
    """Thêm nhiễu Gaussian nhỏ vào tọa độ."""
    coords, presence = _split_processed_sequence(seq)
    coords = coords + np.random.normal(0, sigma, coords.shape)
    return _rebuild_processed_sequence(coords, presence)


def temporal_crop(seq, crop_ratio=0.8):
    """Cắt ngẫu nhiên sequence, pad bằng frame cuối."""
    total = len(seq)
    keep = int(total * crop_ratio)
    start = np.random.randint(0, total - keep + 1)
    cropped = seq[start:start + keep]
    pad = np.tile(cropped[-1], (total - keep, 1))
    return np.concatenate([cropped, pad], axis=0).astype(np.float32)


def scale_variation(seq, scale_range=(0.9, 1.1)):
    """Nhân toàn bộ tọa độ với hệ số scale ngẫu nhiên."""
    scale = np.random.uniform(*scale_range)
    coords, presence = _split_processed_sequence(seq)
    coords = coords * scale
    return _rebuild_processed_sequence(coords, presence)


def time_warp(seq, sigma=0.2):
    """Biến dạng thời gian ngẫu nhiên — co giãn không đều."""
    n = len(seq)
    warp = np.cumsum(np.abs(np.random.normal(1, sigma, n)))
    warp = warp / warp[-1] * (n - 1)
    warp = np.clip(warp, 0, n - 1)
    f = interp1d(np.arange(n), seq, axis=0)
    return f(warp).astype(np.float32)


def rotation_2d(seq, max_angle=15):
    """Xoay nhe tọa độ x,y — mô phỏng nghiêng tay."""
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    coords, presence = _split_processed_sequence(seq)
    rotated = coords.copy()
    num_points = COORD_FEATURES // 3
    for i in range(num_points):
        x, y = coords[:, i * 3], coords[:, i * 3 + 1]
        rotated[:, i * 3] = cos_a * x - sin_a * y
        rotated[:, i * 3 + 1] = sin_a * x + cos_a * y
    return _rebuild_processed_sequence(rotated, presence)


def augment_train():
    """Augment train set: mỗi sample sinh thêm 10 bản -> tổng x11."""
    X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))

    print(f"Trước augmentation: {X_train.shape}")

    aug_path = os.path.join(PROCESSED_PATH, 'X_train_aug.npy')
    if os.path.exists(aug_path):
        print(f"\n[WARNING] File {aug_path} đã tồn tại!")
        resp = input("Ghi đè? (y/n): ").strip().lower()
        if resp != 'y':
            print("Huy bo.")
            return

    np.random.seed(42)
    X_aug, y_aug = [], []

    for i in range(len(X_train)):
        seq = X_train[i]
        lbl = y_train[i]

        X_aug.append(seq)                                       
        y_aug.append(lbl)

        X_aug.append(speed_variation(seq, 0.9))               
        y_aug.append(lbl)

        X_aug.append(speed_variation(seq, 1.1))                 
        y_aug.append(lbl)

        X_aug.append(gaussian_jitter(seq, sigma=0.01))          
        y_aug.append(lbl)
        
        # X_aug.append(scale_variation(seq));                      y_aug.append(lbl)

        X_aug.append(rotation_2d(seq, max_angle=3))           
        y_aug.append(lbl)


    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)

    idx = np.random.permutation(len(X_aug))
    X_aug = X_aug[idx]
    y_aug = y_aug[idx]

    np.save(os.path.join(PROCESSED_PATH, 'X_train_aug.npy'), X_aug)
    np.save(os.path.join(PROCESSED_PATH, 'y_train_aug.npy'), y_aug)

    print(f"Sau augmentation: {X_aug.shape}")
    print(f"Tăng từ {len(X_train)} -> {len(X_aug)} samples (x{len(X_aug)//len(X_train)})")
    print("\nĐã lưu:")
    print(f"  X_train_aug.npy : {X_aug.shape}")
    print(f"  y_train_aug.npy : {y_aug.shape}")
    print("  (Val và Test giữ nguyên — không augment)")


if __name__ == '__main__':
    augment_train()
