import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

LANDMARK_DIR = r'C:\AI\Sign-Language-Dectection\datasets\landmarks'
OUTPUT_DIR   = r'C:\AI\Sign-Language-Dectection\datasets\processed'
CSV_PATH     = r'C:\AI\Sign-Language-Dectection\datasets\metadata\dataset.csv'
SEQ_LEN      = 30  # số frame cố định

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df['video_name'] = df['video_name'].str.strip()
df['person']     = df['person'].str.strip().str.lower()
df['label']      = df['label'].str.strip()

#1. Load tất cả landmark 
sequences, labels, persons = [], [], []

for _, row in df.iterrows():
    npy_name = f"{row['id']:04d}_{row['label']}_{row['person']}.npy"
    npy_path = os.path.join(LANDMARK_DIR, npy_name)
    if not os.path.exists(npy_path):
        continue

    seq = np.load(npy_path)  # shape: (N_frames, 63)

    # 2. Normalize theo wrist (landmark 0) 
    wrist  = seq[:, :3]                        # (N, 3)
    origin = np.tile(wrist, (1, 21))           # (N, 63)
    seq    = seq - origin                      # wrist về (0,0,0)

    scale = np.linalg.norm(seq[:, 36:39], axis=1, keepdims=True)  # ngón giữa tip = landmark 12
    scale = np.tile(scale, (1, 63))
    seq   = seq / (scale + 1e-6)              # scale invariant

    #3. Sequence processing — đưa về SEQ_LEN frame 
    N = len(seq)
    if N >= SEQ_LEN:
        # Uniform sampling
        idx = np.linspace(0, N - 1, SEQ_LEN).astype(int)
        seq = seq[idx]
    else:
        # Padding — lặp frame cuối
        pad = np.tile(seq[-1], (SEQ_LEN - N, 1))
        seq = np.concatenate([seq, pad], axis=0)

    sequences.append(seq)
    labels.append(row['label'])
    persons.append(row['person'])

sequences = np.array(sequences)   # (N_samples, 30, 63)
labels    = np.array(labels)
persons   = np.array(persons)

print(f"Tổng sequences: {len(sequences)}")
print(f"Shape: {sequences.shape}")
print(f"Số class: {len(np.unique(labels))}")
print(f"Classes: {sorted(np.unique(labels))}")

#4. Encode label 
le = LabelEncoder()
y  = le.fit_transform(labels)
print(f"\nLabel encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {i:2d} → {cls}")

# 5. Split theo người — subject independent 
# Train: huyen (258 vid) | Val: kieu (186 vid) | Test: thu (135 vid)
train_idx = np.where((persons == 'huyen'))[0]
val_idx   = np.where((persons == 'kieu') | (persons == 'kiều'))[0]
test_idx  = np.where(persons == 'thu')[0]

X_train = sequences[train_idx]
y_train = y[train_idx]
X_val   = sequences[val_idx]
y_val   = y[val_idx]
X_test  = sequences[test_idx]
y_test  = y[test_idx]

print(f"\nSplit theo người:")
print(f"  Train (huyền) : {len(X_train)} samples")
print(f"  Val   (kiều)  : {len(X_val)} samples")
print(f"  Test  (thu)   : {len(X_test)} samples")

# Lưu ra file 
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),   X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),   y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test)

with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print(f"\nĐã lưu vào: {OUTPUT_DIR}")
print("  X_train.npy, y_train.npy")
print("  X_val.npy,   y_val.npy")
print("  X_test.npy,  y_test.npy")
print("  label_encoder.pkl")