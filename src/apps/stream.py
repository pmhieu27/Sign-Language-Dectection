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

from src.config import MODEL_FACE_LANDMARK_PATH, MODEL_LANDMARK_PATH, SEQ_LEN
from src.data.preprocess import normalize_sequence
from src.inference.pipeline import (
    build_hand_blocks,
    create_face_landmarker,
    create_hand_landmarker,
    draw_hands,
    extract_face_anchors,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition via Webcam")
    parser.add_argument('--model_path', type=str, required=True, help='Duong dan toi file model (.keras)')
    parser.add_argument('--label_encoder', type=str, default=None, help='Duong dan file label encoder (.pkl)')
    parser.add_argument('--camera_id', type=int, default=0, help='ID cua Webcam')
    args = parser.parse_args()

    if args.label_encoder is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    VisionRunningMode = mp.tasks.vision.RunningMode

    latest_result = None

    def landmarker_callback(result, output_image, timestamp_ms):
        nonlocal latest_result
        latest_result = result

    hand_landmarker = create_hand_landmarker(
        VisionRunningMode.LIVE_STREAM,
        result_callback=landmarker_callback,
    )
    face_landmarker = create_face_landmarker(VisionRunningMode.VIDEO)

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
