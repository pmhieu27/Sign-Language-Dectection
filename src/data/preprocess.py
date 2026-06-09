"""
Step 3: Normalize landmarks + pad/sample + split theo person (subject-independent)

Usage: python src/preprocess.py
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.preprocessing import LabelEncoder

from src.config import (
    LANDMARK_PATH, METADATA_PATH, PROCESSED_PATH,
    SEQ_LEN, NUM_FEATURES, NUM_HANDS, NUM_HAND_LANDMARKS, NUM_FACE_LANDMARKS,
    HAND_COORD_FEATURES, FACE_COORD_FEATURES, COORD_FEATURES,
    PRESENCE_FEATURES, RAW_FEATURES, INTER_HAND_FEATURES,
    TRAIN_PERSONS, VAL_PERSONS, TEST_PERSONS,
    SELECTED_CLASSES
)


LEFT_HAND_SLICE = slice(0, NUM_HAND_LANDMARKS * 3)
RIGHT_HAND_SLICE = slice(NUM_HAND_LANDMARKS * 3, NUM_HAND_LANDMARKS * 6)
FACE_SLICE = slice(HAND_COORD_FEATURES, HAND_COORD_FEATURES + FACE_COORD_FEATURES)
PRESENCE_SLICE = slice(COORD_FEATURES, COORD_FEATURES + PRESENCE_FEATURES)
INTER_HAND_SLICE = slice(RAW_FEATURES, NUM_FEATURES)
PALM_INDICES = [0, 5, 9, 13, 17]
TIP_INDICES = [4, 8, 12, 16, 20]


def _reshape_hand(block):
    return block.reshape(len(block), NUM_HAND_LANDMARKS, 3)


def compute_inter_hand_features(left_block, right_block, presence):
    left = _reshape_hand(left_block)
    right = _reshape_hand(right_block)

    left_wrist = left[:, 0, :]
    right_wrist = right[:, 0, :]
    wrist_vector = right_wrist - left_wrist
    wrist_distance = np.linalg.norm(wrist_vector, axis=1, keepdims=True)

    left_palm = left[:, PALM_INDICES, :].mean(axis=1)
    right_palm = right[:, PALM_INDICES, :].mean(axis=1)
    palm_vector = right_palm - left_palm
    palm_distance = np.linalg.norm(palm_vector, axis=1, keepdims=True)

    tip_distances = []
    for idx in TIP_INDICES:
        tip_dist = np.linalg.norm(right[:, idx, :] - left[:, idx, :], axis=1, keepdims=True)
        tip_distances.append(tip_dist)

    inter = np.concatenate(
        [wrist_vector, wrist_distance, palm_vector, palm_distance] + tip_distances,
        axis=1
    )

    both_hands = (presence[:, 0:1] * presence[:, 1:2]).astype(np.float32)
    return inter * both_hands


def build_final_features(seq):
    coords = seq[:, :COORD_FEATURES]
    presence = seq[:, PRESENCE_SLICE]

    left_block = coords[:, LEFT_HAND_SLICE]
    right_block = coords[:, RIGHT_HAND_SLICE]
    face_block = coords[:, FACE_SLICE]
    inter = compute_inter_hand_features(left_block, right_block, presence)

    return np.concatenate([coords, presence, inter], axis=1).astype(np.float32)


def normalize_sequence(seq):
    """
    Normalize landmark sequence:
    Mức 2 (2 tay + mat):
    1. Đổi gốc toạ độ về mũi
    2. Scale theo khoảng cách 2 mắt
    3. Giữ presence mask và thêm inter-hand features
    """
    seq = seq.copy()
    coords = seq[:, :COORD_FEATURES]
    presence = seq[:, PRESENCE_SLICE]

    nose_start = HAND_COORD_FEATURES
    left_eye_start = HAND_COORD_FEATURES + 3
    right_eye_start = HAND_COORD_FEATURES + 6

    num_points = COORD_FEATURES // 3
    origin = np.tile(coords[:, nose_start:nose_start + 3], (1, num_points))
    coords = coords - origin

    eye_vector = coords[:, left_eye_start:left_eye_start + 3] - coords[:, right_eye_start:right_eye_start + 3]
    scale = np.linalg.norm(eye_vector, axis=1, keepdims=True)
    scale = np.tile(scale, (1, COORD_FEATURES))
    coords = coords / (scale + 1e-6)

    seq[:, :COORD_FEATURES] = coords
    seq[:, PRESENCE_SLICE] = presence
    return build_final_features(seq)


def pad_or_sample(seq, target_len):
    """Dua sequence ve chieu dai co dinh."""
    n = len(seq)
    if n >= target_len:
        # Uniform sampling: lay target_len frame deu nhau
        idx = np.linspace(0, n - 1, target_len).astype(int)
        return seq[idx]
    else:
        # Padding: lap frame cuoi
        pad = np.tile(seq[-1], (target_len - n, 1))
        return np.concatenate([seq, pad], axis=0)


def preprocess():
    """Load landmarks, normalize, pad, split va luu."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)

    # Lọc chỉ giữ các class được chọn
    if SELECTED_CLASSES:
        df = df[df['label'].isin(SELECTED_CLASSES)].reset_index(drop=True)
        print(f"Lọc {len(SELECTED_CLASSES)} class: {len(df)} videos")

    sequences, labels, persons = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        npy_name = f"{row['id']:04d}_{row['label']}_{row['person']}.npy"
        npy_path = os.path.join(LANDMARK_PATH, npy_name)

        if not os.path.exists(npy_path):
            skipped += 1
            continue

        seq = np.load(npy_path)
        if seq.ndim != 2 or seq.shape[1] != RAW_FEATURES:
            skipped += 1
            continue

        # Normalize
        seq = normalize_sequence(seq)

        # Pad hoac sample ve SEQ_LEN
        seq = pad_or_sample(seq, SEQ_LEN)

        sequences.append(seq)
        labels.append(row['label'])
        persons.append(row['person'])

    sequences = np.array(sequences, dtype=np.float32)
    labels    = np.array(labels)
    persons   = np.array(persons)

    print(f"Tổng sequences: {len(sequences)}")
    print(f"Shape: {sequences.shape}")
    print(f"So class: {len(np.unique(labels))}")
    if skipped:
        print(f"Skipped (khong tim thay landmark): {skipped}")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"\nLabel encoding:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i:2d} -> {cls}")

    # Subject-independent split
    train_idx = np.isin(persons, TRAIN_PERSONS)
    val_idx   = np.isin(persons, VAL_PERSONS)
    test_idx  = np.isin(persons, TEST_PERSONS)

    X_train, y_train = sequences[train_idx], y[train_idx]
    X_val,   y_val   = sequences[val_idx],   y[val_idx]
    X_test,  y_test  = sequences[test_idx],  y[test_idx]

    print(f"\nSubject-independent split:")
    print(f"  Train ({', '.join(TRAIN_PERSONS)}): {len(X_train)} samples")
    print(f"  Val   ({', '.join(VAL_PERSONS)}):   {len(X_val)} samples")
    print(f"  Test  ({', '.join(TEST_PERSONS)}):  {len(X_test)} samples")

    # Luu split
    np.save(os.path.join(PROCESSED_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_PATH, 'X_val.npy'),   X_val)
    np.save(os.path.join(PROCESSED_PATH, 'y_val.npy'),   y_val)
    np.save(os.path.join(PROCESSED_PATH, 'X_test.npy'),  X_test)
    np.save(os.path.join(PROCESSED_PATH, 'y_test.npy'),  y_test)

    # Luu ALL (cho LOSO cross-validation)
    np.save(os.path.join(PROCESSED_PATH, 'X_all.npy'), sequences)
    np.save(os.path.join(PROCESSED_PATH, 'y_all.npy'), y)
    np.save(os.path.join(PROCESSED_PATH, 'persons_all.npy'), persons)

    with open(os.path.join(PROCESSED_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print(f"\nĐã lưu vào: {PROCESSED_PATH}")


if __name__ == '__main__':
    preprocess()
