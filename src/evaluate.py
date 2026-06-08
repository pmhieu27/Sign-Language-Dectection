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

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix


def get_dirs():
    parser = argparse.ArgumentParser(description='Evaluate Sign Language models')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, 'datasets', 'processed')
    model_dir = args.model_dir or os.path.join(base, 'models')

    return data_dir, model_dir


def evaluate_model(model, model_name, X_test, y_test, label_names):
    """Danh gia 1 model tren test set."""
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

    # Tim cac class sai nhieu nhat
    cm = confusion_matrix(y_test, y_pred, labels=test_labels)
    print(f"  Top confused classes:")
    for i in range(len(test_labels)):
        wrong = cm[i].sum() - cm[i][i]
        if wrong > 0:
            # Tim class bi nham nhieu nhat
            cm_row = cm[i].copy()
            cm_row[i] = 0  # bo dung
            worst_j = np.argmax(cm_row)
            print(f"    {target_names[i]:15s} -> nham thanh {target_names[worst_j]:15s} ({cm_row[worst_j]}/{cm[i].sum()} lan)")

    return acc, y_pred


def main():
    data_dir, model_dir = get_dirs()

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
        acc, _ = evaluate_model(lstm_model, 'LSTM', X_test, y_test, label_names)
        results['LSTM'] = acc
    else:
        print(f"\n[SKIP] LSTM model not found: {lstm_path}")

    # CNN1D + LSTM
    cnn_lstm_path = os.path.join(model_dir, 'cnn_lstm_best.keras')
    if os.path.exists(cnn_lstm_path):
        cnn_lstm_model = keras.models.load_model(cnn_lstm_path)
        acc, _ = evaluate_model(cnn_lstm_model, 'CNN1D + LSTM', X_test, y_test, label_names)
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
