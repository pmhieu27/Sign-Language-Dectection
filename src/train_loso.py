"""
LOSO Cross-Validation (Leave-One-Subject-Out)
Mỗi fold: 1 người test, 1 người val, còn lại train.
Chạy trên Colab T4.

Usage: python src/train_loso.py --data_dir /content/drive/MyDrive/sign_language/processed
"""

import argparse
import os
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
# pyrefly: ignore [missing-import]
from src.augmentation import (
    speed_variation, gaussian_jitter, temporal_crop,
    scale_variation, time_warp, rotation_2d
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.data_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.data_dir = os.path.join(base, 'datasets', 'processed')
    return args


def augment_data(X, y, factor=11):
    """Augment trực tiếp trong memory — không lưu file."""
    np.random.seed(42)
    X_aug, y_aug = [], []

    for seq, lbl in zip(X, y):
        X_aug.append(seq);                                       y_aug.append(lbl)
        X_aug.append(speed_variation(seq, 0.8));                 y_aug.append(lbl)
        X_aug.append(speed_variation(seq, 1.2));                 y_aug.append(lbl)
        X_aug.append(gaussian_jitter(seq, sigma=0.01));          y_aug.append(lbl)
        X_aug.append(temporal_crop(seq, crop_ratio=0.8));        y_aug.append(lbl)
        X_aug.append(scale_variation(seq));                      y_aug.append(lbl)
        X_aug.append(time_warp(seq, sigma=0.2));                 y_aug.append(lbl)
        X_aug.append(rotation_2d(seq, max_angle=15));            y_aug.append(lbl)
        X_aug.append(gaussian_jitter(seq, sigma=0.02));          y_aug.append(lbl)
        X_aug.append(speed_variation(seq, 0.7));                 y_aug.append(lbl)
        X_aug.append(scale_variation(gaussian_jitter(seq, 0.01))); y_aug.append(lbl)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


def build_cnn_lstm(input_shape, num_classes):
    """CNN1D + LSTM model."""
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])


def main():
    args = parse_args()

    # Load data
    X_all = np.load(os.path.join(args.data_dir, 'X_all.npy'))
    y_all = np.load(os.path.join(args.data_dir, 'y_all.npy'))
    persons = np.load(os.path.join(args.data_dir, 'persons_all.npy'))

    with open(os.path.join(args.data_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    unique_persons = sorted(set(persons))
    num_classes = len(le.classes_)
    input_shape = X_all.shape[1:]  # (30, 63)

    print(f"Data: {X_all.shape}")
    print(f"Classes: {num_classes}")
    print(f"Persons: {unique_persons}")
    print(f"\n{'='*60}")
    print(f"  LOSO Cross-Validation ({len(unique_persons)} folds)")
    print(f"{'='*60}\n")

    fold_results = []

    for fold_idx, test_person in enumerate(unique_persons):
        print(f"\n--- Fold {fold_idx+1}/{len(unique_persons)}: Test = {test_person} ---")

        # Split
        test_mask = persons == test_person
        remaining = [p for p in unique_persons if p != test_person]

        # Val = người có ít data nhất trong remaining
        remaining_counts = {p: np.sum(persons == p) for p in remaining}
        val_person = min(remaining_counts, key=remaining_counts.get)
        val_mask = persons == val_person
        train_mask = ~test_mask & ~val_mask

        X_test_fold = X_all[test_mask]
        y_test_fold = y_all[test_mask]
        X_val_fold = X_all[val_mask]
        y_val_fold = y_all[val_mask]
        X_train_fold = X_all[train_mask]
        y_train_fold = y_all[train_mask]

        # Augment train
        X_train_aug, y_train_aug = augment_data(X_train_fold, y_train_fold)

        print(f"  Train: {len(X_train_aug)} (aug from {len(X_train_fold)})")
        print(f"  Val:   {len(X_val_fold)} ({val_person})")
        print(f"  Test:  {len(X_test_fold)}")

        # Class weights
        cw = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
        cw_dict = dict(enumerate(cw))

        # Build & train
        model = build_cnn_lstm(input_shape, num_classes)
        model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        ]

        model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val_fold, y_val_fold),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=0
        )

        # Evaluate
        y_pred = model.predict(X_test_fold, verbose=0).argmax(axis=1)
        acc = np.mean(y_pred == y_test_fold)
        fold_results.append(acc)

        print(f"  ✓ Accuracy: {acc:.4f} ({acc*100:.1f}%)")

        # Cleanup để giải phóng GPU memory
        del model
        tf.keras.backend.clear_session()

    # ===== KẾT QUẢ TỔNG =====
    accs = np.array(fold_results)
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ LOSO ({len(unique_persons)} folds)")
    print(f"{'='*60}")
    for i, (person, acc) in enumerate(zip(unique_persons, fold_results)):
        bar = '█' * int(acc * 30)
        print(f"  Fold {i+1} ({person}): {acc*100:5.1f}% |{bar}|")
    print(f"{'='*60}")
    print(f"  Trung bình:  {accs.mean()*100:.1f}%")
    print(f"  Độ lệch:     ±{accs.std()*100:.1f}%")
    print(f"  Min:          {accs.min()*100:.1f}%")
    print(f"  Max:          {accs.max()*100:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
