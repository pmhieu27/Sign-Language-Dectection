"""Shared runtime helpers for webcam and Streamlit inference."""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

from src.config import (
    FACE_ANCHOR_INDICES,
    MODEL_FACE_LANDMARK_PATH,
    MODEL_LANDMARK_PATH,
    NUM_HAND_LANDMARKS,
    SEQ_LEN,
)
from src.data.preprocess import normalize_sequence, pad_or_sample


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (13, 17),
]


@dataclass
class RuntimeAssets:
    model: keras.Model
    label_encoder: object
    model_path: str
    label_encoder_path: str


def extract_face_anchors(face_landmarks):
    return np.array(
        [[face_landmarks[idx].x, face_landmarks[idx].y, face_landmarks[idx].z]
         for idx in FACE_ANCHOR_INDICES],
        dtype=np.float32,
    ).flatten()


def build_hand_blocks(result):
    left_block = np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
    right_block = np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
    presence = np.zeros(2, dtype=np.float32)
    left_landmarks = None
    right_landmarks = None

    if not result or not result.hand_landmarks:
        return left_block, right_block, presence, left_landmarks, right_landmarks

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        hand_vector = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        ).flatten()

        handedness_list = result.handedness[idx] if idx < len(result.handedness) else []
        handedness = handedness_list[0].category_name if handedness_list else None
        handedness = (handedness or "").lower()

        if handedness == "left":
            left_block = hand_vector
            presence[0] = 1.0
            left_landmarks = hand_landmarks
        elif handedness == "right":
            right_block = hand_vector
            presence[1] = 1.0
            right_landmarks = hand_landmarks
        elif presence[0] == 0.0:
            left_block = hand_vector
            presence[0] = 1.0
            left_landmarks = hand_landmarks
        else:
            right_block = hand_vector
            presence[1] = 1.0
            right_landmarks = hand_landmarks

    return left_block, right_block, presence, left_landmarks, right_landmarks


def draw_hands(frame, hands_to_draw, width, height):
    hand_colors = {
        "LEFT": (0, 255, 0),
        "RIGHT": (255, 0, 0),
    }

    for hand_name, one_hand in hands_to_draw:
        color = hand_colors.get(hand_name, (0, 255, 255))
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_lm = one_hand[start_idx]
            end_lm = one_hand[end_idx]
            start_pt = (int(start_lm.x * width), int(start_lm.y * height))
            end_pt = (int(end_lm.x * width), int(end_lm.y * height))
            cv2.line(frame, start_pt, end_pt, color, 2)

        for lm in one_hand:
            pt = (int(lm.x * width), int(lm.y * height))
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        wrist = one_hand[0]
        wrist_pt = (int(wrist.x * width), int(wrist.y * height))
        cv2.putText(
            frame,
            hand_name,
            (wrist_pt[0] + 8, wrist_pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def create_hand_landmarker(running_mode, result_callback=None):
    base_options = mp.tasks.BaseOptions
    hand_landmarker = mp.tasks.vision.HandLandmarker
    hand_options = mp.tasks.vision.HandLandmarkerOptions
    options = hand_options(
        base_options=base_options(model_asset_path=MODEL_LANDMARK_PATH),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )
    return hand_landmarker.create_from_options(options)


def create_face_landmarker(running_mode):
    base_options = mp.tasks.BaseOptions
    face_landmarker = mp.tasks.vision.FaceLandmarker
    face_options = mp.tasks.vision.FaceLandmarkerOptions
    options = face_options(
        base_options=base_options(model_asset_path=MODEL_FACE_LANDMARK_PATH),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return face_landmarker.create_from_options(options)


def predict_from_sequence(sequence, assets):
    if sequence is None or len(sequence) == 0:
        return None

    raw_seq = np.asarray(sequence, dtype=np.float32)
    normalized = normalize_sequence(raw_seq)
    normalized = pad_or_sample(normalized, SEQ_LEN)
    input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
    predictions = assets.model.predict(input_data, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    return {
        "label": assets.label_encoder.classes_[best_index],
        "confidence": float(predictions[best_index]),
        "probabilities": predictions,
    }
