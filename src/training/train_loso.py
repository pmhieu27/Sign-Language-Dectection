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

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from src.data.augmentation import (
    speed_variation, gaussian_jitter, temporal_crop,
    scale_variation, time_warp, rotation_2d
)
from src.training.models import build_cnn_lstm_model

FACTOR = 5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.data_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.data_dir = os.path.join(base, 'datasets', 'processed')
    return args


def set_seed(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def augment_data(X, y, seed=123):
    """Augment trực tiếp trong memory — không lưu file.
    Comment/uncomment từng dòng augmentation bên dưới để tuỳ chỉnh.
    
    Args:
        seed: Random seed (nên khác nhau cho mỗi fold).
    """
    rng = np.random.default_rng(seed)
    X_aug, y_aug = [], []

    for seq, lbl in zip(X, y):
        X_aug.append(seq)
        y_aug.append(lbl)

        X_aug.append(speed_variation(seq, 0.9))
        y_aug.append(lbl)

        X_aug.append(speed_variation(seq, 1.1))
        y_aug.append(lbl)

        #nhiễu
        X_aug.append(gaussian_jitter(seq, sigma=0.02))
        y_aug.append(lbl)

        X_aug.append(rotation_2d(seq, max_angle=3))
        y_aug.append(lbl)


    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    idx = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]

def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data
    X_all = np.load(os.path.join(args.data_dir, 'X_all.npy'))
    y_all = np.load(os.path.join(args.data_dir, 'y_all.npy'))
    persons = np.load(os.path.join(args.data_dir, 'persons_all.npy'))

    with open(os.path.join(args.data_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    # Chỉ lấy 5 người cần dùng
    SELECTED_PERSONS = ['person_01', 'person_02', 'person_03', 'person_07', 'person_08']
    mask = np.isin(persons, SELECTED_PERSONS)
    X_all = X_all[mask]
    y_all = y_all[mask]
    persons = persons[mask]

    unique_persons = sorted(set(persons))
    num_classes = len(le.classes_)
    input_shape = X_all.shape[1:]

    print(f"Data: {X_all.shape}")
    print(f"Classes: {num_classes}")
    print(f"Persons: {unique_persons}")
    print(f"Seed: {args.seed}")
    print(f"\n{'='*60}")
    print(f"  LOSO Cross-Validation ({len(unique_persons)} folds)")
    print(f"{'='*60}\n")

    fold_results = []

    for fold_idx, test_person in enumerate(unique_persons):
        fold_seed = args.seed + fold_idx
        set_seed(fold_seed)
        print(f"\n--- Fold {fold_idx+1}/{len(unique_persons)}: Test = {test_person} ---")
        print(f"  Seed:  {fold_seed}")

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
        X_train_aug, y_train_aug = augment_data(X_train_fold, y_train_fold, seed=fold_seed)

        print(f"  Train: {len(X_train_aug)} (aug from {len(X_train_fold)})")
        print(f"  Val:   {len(X_val_fold)} ({val_person})")
        print(f"  Test:  {len(X_test_fold)}")

        # Class weights
        unique_classes = np.unique(y_train_aug)
        cw = compute_class_weight('balanced', classes=unique_classes, y=y_train_aug)
        cw_dict = dict(zip(unique_classes, cw))

        # Build & train
        model = build_cnn_lstm_model(input_shape[0], input_shape[1], num_classes, name='CNN1D_LSTM_LOSO')
        model.compile(
            optimizer=Adam(learning_rate=7e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        ckpt_path = os.path.join(args.data_dir, f'loso_fold{fold_idx}_temp.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=False, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
            ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0),
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

        # Load best checkpoint và evaluate
        model.load_weights(ckpt_path)
        y_pred = model.predict(X_test_fold, verbose=0).argmax(axis=1)
        acc = np.mean(y_pred == y_test_fold)
        fold_results.append(acc)

        # Xóa file tạm
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

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
