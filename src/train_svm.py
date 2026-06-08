"""
SVM Classifier cho Sign Language — LOSO Cross-Validation
Không cần GPU, chạy nhanh trên Colab CPU.

Usage: python src/train_svm.py --data_dir /content/drive/MyDrive/sign_language/processed
"""

import argparse
import os
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.data_dir = os.path.join(base, 'datasets', 'processed')
    return args


def extract_features(X):
    """
    Trích xuất features thống kê từ sequence (N, 30, 63) → (N, 441)

    Cho mỗi chiều (63 features):
      - mean, std, min, max          → 63 × 4 = 252
      - delta mean, delta std        → 63 × 2 = 126  (vận tốc)
      - đầu-cuối difference          → 63 × 1 = 63   (biến đổi tổng)
    Tổng: 441 features
    """
    feats = []
    for seq in X:
        # Thống kê cơ bản (252)
        f_mean = seq.mean(axis=0)
        f_std  = seq.std(axis=0)
        f_min  = seq.min(axis=0)
        f_max  = seq.max(axis=0)

        # Vận tốc: sai phân giữa các frame (126)
        delta = np.diff(seq, axis=0)  # (29, 63)
        d_mean = delta.mean(axis=0)
        d_std  = delta.std(axis=0)

        # Biến đổi tổng: frame cuối - frame đầu (63)
        diff = seq[-1] - seq[0]

        feat = np.concatenate([f_mean, f_std, f_min, f_max, d_mean, d_std, diff])
        feats.append(feat)

    return np.array(feats)


def main():
    args = parse_args()

    # Load data
    X_all   = np.load(os.path.join(args.data_dir, 'X_all.npy'))
    y_all   = np.load(os.path.join(args.data_dir, 'y_all.npy'))
    persons = np.load(os.path.join(args.data_dir, 'persons_all.npy'))

    with open(os.path.join(args.data_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    unique_persons = sorted(set(persons))
    label_names = le.classes_

    print(f"Data: {X_all.shape}")
    print(f"Classes: {len(label_names)}")
    print(f"Persons: {unique_persons}")

    # Trích xuất features
    print("\nĐang trích xuất features...")
    X_feat = extract_features(X_all)
    print(f"Feature shape: {X_feat.shape}")

    # ===== LOSO Cross-Validation =====
    print(f"\n{'='*60}")
    print(f"  SVM + LOSO Cross-Validation ({len(unique_persons)} folds)")
    print(f"{'='*60}")

    fold_accs = []
    all_y_true, all_y_pred = [], []

    for fold_idx, test_person in enumerate(unique_persons):
        test_mask  = persons == test_person
        train_mask = ~test_mask

        X_train = X_feat[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_feat[test_mask]
        y_test  = y_all[test_mask]

        # Pipeline: StandardScaler → SVM (RBF kernel)
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                decision_function_shape='ovr'
            ))
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold_idx+1} ({test_person}): {acc*100:5.1f}%  "
              f"({np.sum(y_pred == y_test)}/{len(y_test)})")

    # ===== KẾT QUẢ =====
    accs = np.array(fold_accs)
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ SVM + LOSO")
    print(f"{'='*60}")
    print(f"  Trung bình:  {accs.mean()*100:.1f}%")
    print(f"  Độ lệch:     ±{accs.std()*100:.1f}%")
    print(f"  Min:          {accs.min()*100:.1f}%  (Fold {accs.argmin()+1})")
    print(f"  Max:          {accs.max()*100:.1f}%  (Fold {accs.argmax()+1})")
    print(f"{'='*60}")

    # Classification report tổng hợp
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    test_labels = sorted(set(all_y_true))
    target_names = [label_names[i] for i in test_labels]

    print(f"\n  Classification Report (tổng hợp tất cả fold):")
    print(classification_report(all_y_true, all_y_pred,
                                labels=test_labels,
                                target_names=target_names,
                                zero_division=0))


if __name__ == '__main__':
    main()
