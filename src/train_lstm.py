"""
Step 5a: Train LSTM model
Chay tren Colab T4: python src/train_lstm.py --data_dir /content/drive/MyDrive/sign_language/processed

Usage (local):  python src/train_lstm.py
Usage (Colab):  python src/train_lstm.py --data_dir /content/drive/MyDrive/sign_language/processed
"""

import argparse
import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


def get_data_dir():
    """Xac dinh thu muc du lieu tu argument hoac mac dinh."""
    parser = argparse.ArgumentParser(description='Train LSTM for Sign Language')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to processed data directory')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to save model')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Try local path
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base, 'datasets', 'processed')

    if args.model_dir:
        model_dir = args.model_dir
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base, 'models')

    return data_dir, model_dir, args


def build_lstm_model(seq_len, num_features, num_classes):
    """Xay d 32 ung LSTM model."""
    model = keras.Sequential([
        layers.Input(shape=(seq_len, num_features)),

        # LSTM layers
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.LSTM(64),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Dense classifier
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ], name='LSTM_SignLanguage')

    return model


def main():
    data_dir, model_dir, args = get_data_dir()
    os.makedirs(model_dir, exist_ok=True)

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs: {gpus}")
    if gpus:
        print(f"Training on GPU: {gpus[0].name}")
    else:
        print("Training on CPU")

    # Load data
    print(f"\nLoading data from: {data_dir}")
    X_train = np.load(os.path.join(data_dir, 'X_train_aug.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_aug.npy'))
    X_val   = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val   = np.load(os.path.join(data_dir, 'y_val.npy'))

    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]

    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"Classes: {num_classes}")

    # Class weights (xu ly imbalance)
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights (min={min(class_weights):.2f}, max={max(class_weights):.2f})")

    # Build model
    model = build_lstm_model(seq_len, num_features, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    model_path = os.path.join(model_dir, 'lstm_best.keras')
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    # Train
    print(f"\nTraining LSTM (epochs={args.epochs}, batch_size={args.batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    hist_path = os.path.join(model_dir, 'lstm_history.json')
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, 'w') as f:
        json.dump(hist_data, f)

    # Final report
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    print(f"\n{'='*50}")
    print(f"LSTM TRAINING COMPLETE")
    print(f"  Best val_accuracy: {best_val_acc:.4f}")
    print(f"  Best val_loss:     {best_val_loss:.4f}")
    print(f"  Model saved:       {model_path}")
    print(f"  History saved:     {hist_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
