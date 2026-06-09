"""
Step 6: Danh gia va so sanh 2 model tren test set

Usage (local):  python src/evaluate.py
Usage (Colab):  python src/evaluate.py --data_dir /path/to/processed --model_dir /path/to/models
"""

import argparse
import os
import numpy as np
import pickle
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — hoat dong tren server/Colab
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix


def get_dirs():
    parser = argparse.ArgumentParser(description='Evaluate Sign Language models')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Thu muc luu confusion matrix plots. Mac dinh: models/')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = args.data_dir or os.path.join(base, 'datasets', 'processed')
    model_dir = args.model_dir or os.path.join(base, 'models')
    save_dir = args.save_dir or model_dir

    return data_dir, model_dir, save_dir


def plot_confusion_matrix(cm, target_names, model_name, save_path, normalize=False):
    """Ve confusion matrix heatmap va luu thanh file PNG.

    Args:
        cm: Confusion matrix (numpy array).
        target_names: Danh sach ten label.
        model_name: Ten model (de hien thi tren title).
        save_path: Duong dan luu file PNG.
        normalize: Neu True, hien thi % thay vi so luong.
    """
    if normalize:
        # Normalize theo tung row (true label) -> ty le %
        cm_display = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
        fmt = '.1f'
        value_suffix = '%'
        title_suffix = ' (Normalized %)'
    else:
        cm_display = cm.astype(np.float64)
        fmt = '.0f'
        value_suffix = ''
        title_suffix = ' (Counts)'

    n_classes = len(target_names)
    fig_size = max(6, n_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))

    # Color map
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Ticks & labels
    tick_positions = np.arange(n_classes)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(target_names, fontsize=9)

    # Ghi so len tung o
    thresh = cm_display.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_display[i, j]
            text = f"{value:{fmt}}{value_suffix}"
            color = 'white' if value > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}{title_suffix}', fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {save_path}")


def evaluate_model(model, model_name, X_test, y_test, label_names, save_dir=None):
    """Danh gia 1 model tren test set va ve confusion matrix."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    # Predict
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Overall accuracy
    acc = np.mean(y_pred == y_test)
    print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Xac dinh cac label co mat trong test set
    test_labels = sorted(set(y_test))
    target_names = [label_names[i] for i in test_labels]

    # Classification report
    print(f"\n  Classification Report:")
    report = classification_report(
        y_test, y_pred,
        labels=test_labels,
        target_names=target_names,
        zero_division=0
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=test_labels)

    # Tim cac class sai nhieu nhat
    print(f"  Top confused classes:")
    for i in range(len(test_labels)):
        wrong = cm[i].sum() - cm[i][i]
        if wrong > 0:
            # Tim class bi nham nhieu nhat
            cm_row = cm[i].copy()
            cm_row[i] = 0  # bo dung
            worst_j = np.argmax(cm_row)
            print(f"    {target_names[i]:15s} -> nham thanh {target_names[worst_j]:15s} ({cm_row[worst_j]}/{cm[i].sum()} lan)")

    # Ve confusion matrix heatmap
    if save_dir:
        safe_name = model_name.lower().replace(' ', '_').replace('+', '_')

        # Raw counts
        count_path = os.path.join(save_dir, f'confusion_matrix_{safe_name}.png')
        plot_confusion_matrix(cm, target_names, model_name, count_path, normalize=False)

        # Normalized %
        norm_path = os.path.join(save_dir, f'confusion_matrix_{safe_name}_normalized.png')
        plot_confusion_matrix(cm, target_names, model_name, norm_path, normalize=True)

    return acc, y_pred


def main():
    data_dir, model_dir, save_dir = get_dirs()
    os.makedirs(save_dir, exist_ok=True)

    # Load test data
    print(f"Loading test data from: {data_dir}")
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    label_names = list(le.classes_)
    print(f"X_test: {X_test.shape}")
    print(f"Classes in test: {len(set(y_test))}/{len(label_names)}")

    # Kiem tra label nao thieu trong test
    all_labels = set(range(len(label_names)))
    test_labels = set(y_test)
    missing = all_labels - test_labels
    if missing:
        missing_names = [label_names[i] for i in missing]
        print(f"\n[WARNING] Labels THIEU trong test set: {missing_names}")
        print(f"  -> Cac label nay se khong duoc danh gia")

    # Load va evaluate tung model
    results = {}

    # LSTM
    lstm_path = os.path.join(model_dir, 'lstm_best.keras')
    if os.path.exists(lstm_path):
        lstm_model = keras.models.load_model(lstm_path)
        acc, _ = evaluate_model(lstm_model, 'LSTM', X_test, y_test, label_names, save_dir)
        results['LSTM'] = acc
    else:
        print(f"\n[SKIP] LSTM model not found: {lstm_path}")

    # CNN1D + LSTM
    cnn_lstm_path = os.path.join(model_dir, 'cnn_lstm_best.keras')
    if os.path.exists(cnn_lstm_path):
        cnn_lstm_model = keras.models.load_model(cnn_lstm_path)
        acc, _ = evaluate_model(cnn_lstm_model, 'CNN1D + LSTM', X_test, y_test, label_names, save_dir)
        results['CNN1D+LSTM'] = acc
    else:
        print(f"\n[SKIP] CNN1D+LSTM model not found: {cnn_lstm_path}")

    # So sanh
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  SO SANH KET QUA")
        print(f"{'='*60}")
        for name, acc in sorted(results.items(), key=lambda x: -x[1]):
            bar = '#' * int(acc * 40)
            print(f"  {name:15s}: {acc:.4f} ({acc*100:.1f}%) |{bar}|")

        best = max(results, key=results.get)
        print(f"\n  -> Model tot nhat: {best} ({results[best]*100:.1f}%)")

if __name__ == '__main__':
    main()
