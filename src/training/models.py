"""Shared training model builders."""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def build_cnn_lstm_model(seq_len, num_features, num_classes, name="CNN1D_LSTM_SignLanguage"):
    return keras.Sequential(
        [
            layers.Input(shape=(seq_len, num_features)),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            # layers.Bidirectional(layers.LSTM(64)),
            layers.LSTM(64),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name=name,
    )


def build_lstm_model(seq_len, num_features, num_classes, name="LSTM_SignLanguage"):
    return keras.Sequential(
        [
            layers.Input(shape=(seq_len, num_features)),
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name=name,
    )
