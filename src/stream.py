"""
Stream webcam thoi gian thuc va nhan dien ngon ngu ky hieu.

Usage:
    python -m src.stream --model_path models/cnn_lstm_best.keras
"""

import argparse
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pickle

os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

# pyrefly: ignore [missing-import]
from setting.config import (
    MODEL_LANDMARK_PATH, MODEL_FACE_LANDMARK_PATH, SEQ_LEN,
    FACE_ANCHOR_INDICES, NUM_HAND_LANDMARKS
)
# pyrefly: ignore [missing-import]
from src.preprocess import normalize_sequence


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (13, 17)
]


def extract_face_anchors(face_landmarks):
    return np.array(
        [[face_landmarks[idx].x, face_landmarks[idx].y, face_landmarks[idx].z]
         for idx in FACE_ANCHOR_INDICES],
        dtype=np.float32
    ).flatten()


def create_face_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FACE_LANDMARK_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


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
            dtype=np.float32
        ).flatten()

        handedness_list = result.handedness[idx] if idx < len(result.handedness) else []
        handedness = handedness_list[0].category_name if handedness_list else None
        handedness = (handedness or '').lower()

        if handedness == 'left':
            left_block = hand_vector
            presence[0] = 1.0
            left_landmarks = hand_landmarks
        elif handedness == 'right':
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
        'LEFT': (0, 255, 0),
        'RIGHT': (255, 0, 0),
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
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

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
            cv2.LINE_AA
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition via Webcam")
    parser.add_argument('--model_path', type=str, required=True, help='Duong dan toi file model (.keras)')
    parser.add_argument('--label_encoder', type=str, default=None, help='Duong dan file label encoder (.pkl)')
    parser.add_argument('--camera_id', type=int, default=0, help='ID cua Webcam')
    args = parser.parse_args()

    if args.label_encoder is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.label_encoder = os.path.join(base, 'datasets', 'processed', 'label_encoder.pkl')

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.label_encoder):
        print(f"Loi: Khong tim thay label encoder tai: {args.label_encoder}")
        return
    if not os.path.exists(args.model_path):
        print(f"Loi: Khong tim thay model tai: {args.model_path}")
        return
    if not os.path.exists(MODEL_LANDMARK_PATH):
        print(f"Loi: Khong tim thay hand landmarker tai: {MODEL_LANDMARK_PATH}")
        return
    if not os.path.exists(MODEL_FACE_LANDMARK_PATH):
        print(f"Loi: Khong tim thay face landmarker tai: {MODEL_FACE_LANDMARK_PATH}")
        return

    with open(args.label_encoder, 'rb') as f:
        le = pickle.load(f)

    model = keras.models.load_model(args.model_path)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    latest_result = None

    def landmarker_callback(result, output_image, timestamp_ms):
        nonlocal latest_result
        latest_result = result

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_LANDMARK_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=landmarker_callback,
    )

    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    face_landmarker = create_face_landmarker()

    sequence_buffer = deque(maxlen=SEQ_LEN)
    latest_face_vector = None
    predicted_label = "..."
    confidence = 0.0
    current_presence = np.zeros(2, dtype=np.float32)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Loi: Khong the mo camera ID={args.camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        hand_landmarker.detect_async(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            latest_face_vector = extract_face_anchors(face_landmarks)
            for idx in FACE_ANCHOR_INDICES:
                lm = face_landmarks[idx]
                pt = (int(lm.x * width), int(lm.y * height))
                cv2.circle(frame, pt, 3, (255, 200, 0), -1)

        if latest_result is not None and latest_face_vector is not None:
            left_block, right_block, presence, left_lms, right_lms = build_hand_blocks(latest_result)
            current_presence = presence
            hands_to_draw = []
            if left_lms is not None:
                hands_to_draw.append(('LEFT', left_lms))
            if right_lms is not None:
                hands_to_draw.append(('RIGHT', right_lms))
            draw_hands(frame, hands_to_draw, width, height)

            if presence.sum() > 0:
                vector = np.concatenate([left_block, right_block, latest_face_vector, presence])
                sequence_buffer.append(vector)
            elif len(sequence_buffer) > 0:
                sequence_buffer.popleft()

            if len(sequence_buffer) == SEQ_LEN:
                raw_seq = np.array(sequence_buffer, dtype=np.float32)
                normalized = normalize_sequence(raw_seq)
                input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
                predictions = model.predict(input_data, verbose=0)[0]
                predicted_idx = int(np.argmax(predictions))
                confidence = float(predictions[predicted_idx])
                predicted_label = le.classes_[predicted_idx]
        elif len(sequence_buffer) > 0:
            sequence_buffer.popleft()

        cv2.rectangle(frame, (0, 0), (640, 55), (0, 0, 0), -1)
        display_text = f"Ky hieu: {predicted_label.upper()} ({confidence * 100:.1f}%)"
        cv2.putText(frame, display_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if confidence > 0.6 else (0, 255, 255), 2, cv2.LINE_AA)
        hand_status = f"Hands: L={int(current_presence[0])} R={int(current_presence[1])}"
        cv2.putText(frame, hand_status, (15, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Nhan 'q' de thoat", (480, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Nhan dien Ngon ngu Ky hieu Real-time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    face_landmarker.close()
    print("Da tat camera.")


if __name__ == '__main__':
    main()
