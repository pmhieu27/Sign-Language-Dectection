"""
Evaluate the final tuned model on the held-out test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model_path models/cnn_lstm_final.keras
"""

import argparse
import json
import os
import pickle

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate final tuned sign language model")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base, "models")
    if args.data_dir is None:
        args.data_dir = os.path.join(base, "datasets", "processed")
    if args.model_path is None:
        args.model_path = os.path.join(model_dir, "cnn_lstm_final.keras")
    if args.save_dir is None:
        args.save_dir = model_dir

    return args


def plot_confusion_matrix(cm, target_names, save_path, normalize=False):
    if normalize:
        cm_display = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
        fmt = ".1f"
        suffix = "%"
        title = "Confusion Matrix - Final Model (Normalized %)"
    else:
        cm_display = cm.astype(np.float64)
        fmt = ".0f"
        suffix = ""
        title = "Confusion Matrix - Final Model (Counts)"

    fig_size = max(6, len(target_names) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(target_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(target_names, fontsize=9)

    threshold = cm_display.max() / 2.0
    for row in range(len(target_names)):
        for col in range(len(target_names)):
            value = cm_display[row, col]
            ax.text(
                col,
                row,
                f"{value:{fmt}}{suffix}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading test data from: {args.data_dir}")
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    with open(os.path.join(args.data_dir, "label_encoder.pkl"), "rb") as handle:
        label_encoder = pickle.load(handle)

    class_names = list(label_encoder.classes_)
    print(f"X_test: {X_test.shape}")
    print(f"Model:  {args.model_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = keras.models.load_model(args.model_path)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_precision = report_dict["weighted avg"]["precision"]
    weighted_recall = report_dict["weighted avg"]["recall"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print("\nPer-class recall:")
    for class_name in class_names:
        class_metrics = report_dict.get(class_name, {})
        recall = class_metrics.get("recall", 0.0)
        support = class_metrics.get("support", 0)
        print(f"  {class_name:15s}: recall={recall:.4f} support={int(support)}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion summary:")
    for row_index, class_name in enumerate(class_names):
        total = int(cm[row_index].sum())
        correct = int(cm[row_index, row_index])
        wrong = total - correct
        if wrong <= 0:
            print(f"  {class_name:15s}: correct={correct}/{total}, top confusion=None")
            continue

        row = cm[row_index].copy()
        row[row_index] = 0
        confused_index = int(np.argmax(row))
        confused_name = class_names[confused_index]
        confused_count = int(row[confused_index])
        print(
            f"  {class_name:15s}: correct={correct}/{total}, "
            f"top confusion={confused_name} ({confused_count})"
        )

    count_path = os.path.join(args.save_dir, "confusion_matrix_final.png")
    norm_path = os.path.join(args.save_dir, "confusion_matrix_final_normalized.png")
    plot_confusion_matrix(cm, class_names, count_path, normalize=False)
    plot_confusion_matrix(cm, class_names, norm_path, normalize=True)

    report_path = os.path.join(args.save_dir, "evaluation_final_report.json")
    summary_path = os.path.join(args.save_dir, "evaluation_final_summary.json")

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, ensure_ascii=False, indent=2)

    summary = {
        "model_path": args.model_path,
        "test_shape": list(X_test.shape),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "num_classes": len(class_names),
        "class_names": class_names,
        "confusion_matrix_path": count_path,
        "confusion_matrix_normalized_path": norm_path,
        "report_path": report_path,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved confusion matrices: {count_path}, {norm_path}")


if __name__ == "__main__":
    main()
