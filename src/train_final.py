"""
Train final CNN1D + LSTM model for the current landmark pipeline.

Default behavior:
- Load `X_train.npy` and `X_val.npy`
- Merge them into one training pool
- Create a fresh internal validation split for callbacks
- Apply light augmentation only on the training subset
- Save the best checkpoint for deployment

Usage:
    python -m src.train_final
    python -m src.train_final --epochs 120 --batch_size 16
    python -m src.train_final --keep-original-val
"""

import argparse
import json
import os
import pickle

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# pyrefly: ignore [missing-import]
from src.augmentation import (
    gaussian_jitter,
    rotation_2d,
    scale_variation,
    speed_variation,
    time_warp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train final CNN1D + LSTM model")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--augment_factor", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--combine-val", dest="combine_val", action="store_true")
    parser.add_argument("--keep-original-val", dest="combine_val", action="store_false")
    parser.set_defaults(combine_val=True)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_dir is None:
        args.data_dir = os.path.join(base, "datasets", "processed")
    if args.model_dir is None:
        args.model_dir = os.path.join(base, "models")

    if not 0 < args.val_ratio < 0.5:
        raise ValueError("--val_ratio must be between 0 and 0.5")
    if args.augment_factor < 1:
        raise ValueError("--augment_factor must be >= 1")

    return args


def set_seed(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_cnn_lstm_model(seq_len, num_features, num_classes):
    return keras.Sequential(
        [
            layers.Input(shape=(seq_len, num_features)),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="CNN1D_LSTM_SignLanguage_Final",
    )


def augmentation_ops():
    return [
        lambda seq: speed_variation(seq, 0.9),
        lambda seq: speed_variation(seq, 1.1),
        lambda seq: gaussian_jitter(seq, sigma=0.01),
        lambda seq: time_warp(seq, sigma=0.12),
        lambda seq: rotation_2d(seq, max_angle=8),
        lambda seq: scale_variation(seq, scale_range=(0.97, 1.03)),
    ]


def augment_dataset(X, y, factor, seed):
    rng = np.random.default_rng(seed)
    ops = augmentation_ops()

    X_aug = []
    y_aug = []

    for seq, label in zip(X, y):
        X_aug.append(seq.astype(np.float32))
        y_aug.append(label)

        for aug_index in range(max(0, factor - 1)):
            op = ops[aug_index % len(ops)]
            X_aug.append(op(seq).astype(np.float32))
            y_aug.append(label)

    X_aug = np.asarray(X_aug, dtype=np.float32)
    y_aug = np.asarray(y_aug)

    order = rng.permutation(len(X_aug))
    return X_aug[order], y_aug[order]


def prepare_splits(args):
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))

    if args.combine_val:
        X_pool = np.concatenate([X_train, X_val], axis=0)
        y_pool = np.concatenate([y_train, y_val], axis=0)
        X_train_base, X_val_internal, y_train_base, y_val_internal = train_test_split(
            X_pool,
            y_pool,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=y_pool,
        )
        split_name = "train+val -> internal validation split"
    else:
        X_train_base, y_train_base = X_train, y_train
        X_val_internal, y_val_internal = X_val, y_val
        split_name = "original train/val split"

    return X_train_base, y_train_base, X_val_internal, y_val_internal, split_name


def class_distribution(y, class_names):
    counts = np.bincount(y, minlength=len(class_names))
    return {class_names[index]: int(count) for index, count in enumerate(counts)}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU devices: {gpus if gpus else 'CPU only'}")
    print(f"Loading data from: {args.data_dir}")

    with open(os.path.join(args.data_dir, "label_encoder.pkl"), "rb") as handle:
        label_encoder = pickle.load(handle)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    X_train_base, y_train_base, X_val_internal, y_val_internal, split_name = prepare_splits(args)
    X_train_aug, y_train_aug = augment_dataset(
        X_train_base, y_train_base, factor=args.augment_factor, seed=args.seed
    )

    seq_len = X_train_aug.shape[1]
    num_features = X_train_aug.shape[2]

    print(f"Split mode: {split_name}")
    print(f"Base train: {X_train_base.shape}")
    print(f"Val used:   {X_val_internal.shape}")
    print(f"Train aug:  {X_train_aug.shape}")
    print(f"Classes:    {num_classes}")
    print(f"Train labels: {class_distribution(y_train_base, class_names)}")
    print(f"Val labels:   {class_distribution(y_val_internal, class_names)}")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_aug),
        y=y_train_aug,
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = build_cnn_lstm_model(seq_len, num_features, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path = os.path.join(args.model_dir, "cnn_lstm_best.keras")
    history_path = os.path.join(args.model_dir, "cnn_lstm_final_history.json")
    metadata_path = os.path.join(args.model_dir, "cnn_lstm_final_metadata.json")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print(
        f"Training final model (epochs={args.epochs}, batch_size={args.batch_size}, "
        f"augment_factor={args.augment_factor})..."
    )
    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val_internal, y_val_internal),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history_data, handle, ensure_ascii=False, indent=2)

    metadata = {
        "split_mode": split_name,
        "combine_val": args.combine_val,
        "val_ratio": args.val_ratio,
        "augment_factor": args.augment_factor,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "base_train_shape": list(X_train_base.shape),
        "train_aug_shape": list(X_train_aug.shape),
        "val_shape": list(X_val_internal.shape),
        "num_classes": num_classes,
        "class_names": class_names,
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "best_val_loss": float(min(history.history["val_loss"])),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print(f"Best val_accuracy: {metadata['best_val_accuracy']:.4f}")
    print(f"Best val_loss:     {metadata['best_val_loss']:.4f}")
    print(f"Model saved:       {model_path}")
    print(f"History saved:     {history_path}")
    print(f"Metadata saved:    {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
