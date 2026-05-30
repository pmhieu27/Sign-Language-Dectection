import numpy as np
import os
from scipy.interpolate import interp1d

PROCESSED_DIR = r'C:\AI\Sign-Language-Dectection\datasets\processed'

# Load data 
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))

print(f"Trước augmentation: {X_train.shape}")

# 4 kỹ thuật augmentation

def speed_variation(seq, factor):
    """Thay đổi tốc độ — tự tìm hiểu"""
    old_len = len(seq)
    new_len = max(2, int(old_len * factor))
    f       = interp1d(np.arange(old_len), seq, axis=0)
    resampled = f(np.linspace(0, old_len - 1, new_len))
    f2      = interp1d(np.arange(new_len), resampled, axis=0)
    return f2(np.linspace(0, new_len - 1, old_len))

def gaussian_jitter(seq, sigma=0.01):
    """Thêm nhiễu nhỏ"""
    return seq + np.random.normal(0, sigma, seq.shape)
def horizontal_flip(seq):
    """Lật ngang — mô phỏng tay trái"""
    flipped = seq.copy()
    # Đảo tọa độ x của tất cả 21 landmark
    # Mỗi landmark có 3 giá trị x,y,z tại index 0,3,6,...,60
    for i in range(21):
        flipped[:, i*3] = -seq[:, i*3]  # đảo x, giữ nguyên y và z
    return flipped
def temporal_crop(seq, crop_ratio=0.8):
    """Cắt chuỗi ngẫu nhiên — tự tìm hiểu"""
    total = len(seq)
    keep  = int(total * crop_ratio)
    start = np.random.randint(0, total - keep)
    cropped = seq[start:start + keep]
    pad   = np.tile(cropped[-1], (total - keep, 1))
    return np.concatenate([cropped, pad], axis=0)

#Áp dụng — mỗi sample sinh thêm 3 bản augmented
np.random.seed(42)

X_aug, y_aug = [], []

for i in range(len(X_train)):
    seq = X_train[i]
    lbl = y_train[i]

    # Bản gốc giữ nguyên
    X_aug.append(seq)
    y_aug.append(lbl)

    # Speed slow (0.8x)
    X_aug.append(speed_variation(seq, 0.8))
    y_aug.append(lbl)

    # Speed fast (1.2x)
    X_aug.append(speed_variation(seq, 1.2))
    y_aug.append(lbl)

    # Gaussian jitter
    X_aug.append(gaussian_jitter(seq, sigma=0.01))
    y_aug.append(lbl)

    # Temporal crop
    X_aug.append(temporal_crop(seq, crop_ratio=0.8))
    y_aug.append(lbl)

    X_aug.append(horizontal_flip(seq))          # lật ngang 
    y_aug.append(lbl)

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

print(f"Sau augmentation:   {X_aug.shape}")
print(f"Tăng từ {len(X_train)} → {len(X_aug)} samples (×{len(X_aug)//len(X_train)})")

# Shuffle
idx = np.random.permutation(len(X_aug))
X_aug = X_aug[idx]
y_aug = y_aug[idx]

# Lưu đè lên X_train 
np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_aug)
np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_aug)

print(f"\nĐã lưu X_train augmented: {X_aug.shape}")
print(f"Val và Test giữ nguyên — không augment")
print(f"\nData cuối cùng:")
print(f"  X_train : {X_aug.shape} ")
print(f"  X_val   : (184, 30, 63) ")
print(f"  X_test  : (136, 30, 63)")
print(f"  label_encoder.pkl")