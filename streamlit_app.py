import os
import pickle
import tempfile
from collections import deque
from dataclasses import dataclass

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow import keras

from setting.config import (
    MODEL_FACE_LANDMARK_PATH,
    MODEL_LANDMARK_PATH,
    SEQ_LEN,
    FACE_ANCHOR_INDICES,
    NUM_HAND_LANDMARKS,
)
from src.preprocess import normalize_sequence, pad_or_sample


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


def ensure_required_files():
    missing = []
    if not os.path.exists(MODEL_LANDMARK_PATH):
        missing.append(MODEL_LANDMARK_PATH)
    if not os.path.exists(MODEL_FACE_LANDMARK_PATH):
        missing.append(MODEL_FACE_LANDMARK_PATH)
    return missing


def create_hand_landmarker(running_mode):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_LANDMARK_PATH),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def create_face_landmarker(running_mode):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FACE_LANDMARK_PATH),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


@st.cache_resource
def load_assets(model_path, label_encoder_path):
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    model = keras.models.load_model(model_path)
    return RuntimeAssets(
        model=model,
        label_encoder=label_encoder,
        model_path=model_path,
        label_encoder_path=label_encoder_path,
    )


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


def extract_sequence_from_video(video_path):
    VisionRunningMode = mp.tasks.vision.RunningMode
    hand_landmarker = create_hand_landmarker(VisionRunningMode.VIDEO)
    face_landmarker = create_face_landmarker(VisionRunningMode.VIDEO)

    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_count = 0
    last_face_vector = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((frame_count / fps) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if face_result.face_landmarks:
            last_face_vector = extract_face_anchors(face_result.face_landmarks[0])

        if last_face_vector is not None:
            left_block, right_block, presence, _, _ = build_hand_blocks(hand_result)
            if presence.sum() > 0:
                vector = np.concatenate([left_block, right_block, last_face_vector, presence])
                sequence.append(vector)

        frame_count += 1

    cap.release()
    hand_landmarker.close()
    face_landmarker.close()
    return np.array(sequence, dtype=np.float32) if sequence else None


class SignVideoProcessor(VideoProcessorBase):
    def __init__(self, assets):
        self.assets = assets
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.hand_landmarker = create_hand_landmarker(VisionRunningMode.VIDEO)
        self.face_landmarker = create_face_landmarker(VisionRunningMode.VIDEO)
        self.sequence_buffer = deque(maxlen=SEQ_LEN)
        self.latest_face_vector = None
        self.predicted_label = "..."
        self.confidence = 0.0
        self.current_presence = np.zeros(2, dtype=np.float32)
        self.frame_index = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        height, width, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(self.frame_index * (1000 / 30.0))
        self.frame_index += 1

        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            self.latest_face_vector = extract_face_anchors(face_landmarks)
            for idx in FACE_ANCHOR_INDICES:
                lm = face_landmarks[idx]
                pt = (int(lm.x * width), int(lm.y * height))
                cv2.circle(image, pt, 3, (255, 200, 0), -1)

        if self.latest_face_vector is not None:
            left_block, right_block, presence, left_lms, right_lms = build_hand_blocks(hand_result)
            self.current_presence = presence
            hands_to_draw = []
            if left_lms is not None:
                hands_to_draw.append(("LEFT", left_lms))
            if right_lms is not None:
                hands_to_draw.append(("RIGHT", right_lms))
            draw_hands(image, hands_to_draw, width, height)

            if presence.sum() > 0:
                vector = np.concatenate([left_block, right_block, self.latest_face_vector, presence])
                self.sequence_buffer.append(vector)
            elif len(self.sequence_buffer) > 0:
                self.sequence_buffer.popleft()

            if len(self.sequence_buffer) == SEQ_LEN:
                result = predict_from_sequence(np.array(self.sequence_buffer, dtype=np.float32), self.assets)
                if result is not None:
                    self.predicted_label = result["label"]
                    self.confidence = result["confidence"]

        cv2.rectangle(image, (0, 0), (700, 90), (0, 0, 0), -1)
        text = f"Ky hieu: {self.predicted_label.upper()} ({self.confidence * 100:.1f}%)"
        cv2.putText(
            image,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if self.confidence > 0.6 else (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        hand_status = f"Hands: L={int(self.current_presence[0])} R={int(self.current_presence[1])}"
        cv2.putText(
            image,
            hand_status,
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def __del__(self):
        try:
            self.hand_landmarker.close()
        except Exception:
            pass
        try:
            self.face_landmarker.close()
        except Exception:
            pass


def render_upload_tab(assets):
    uploaded_file = st.file_uploader(
        "Upload video (.mov, .mp4, .avi)",
        type=["mov", "mp4", "avi"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    st.video(uploaded_file)

    if st.button("Chay nhan dien video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Dang trich xuat landmark va du doan..."):
                sequence = extract_sequence_from_video(tmp_path)
                if sequence is None or len(sequence) < 10:
                    st.error("Khong du frame landmark hop le de du doan.")
                    return
                result = predict_from_sequence(sequence, assets)
                if result is None:
                    st.error("Khong the du doan tren video nay.")
                    return

            st.success("Hoan thanh du doan.")
            st.metric("Nhan du doan", result["label"])
            st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

            class_names = list(assets.label_encoder.classes_)
            top_indices = np.argsort(result["probabilities"])[::-1][:5]
            st.subheader("Top 5 du doan")
            for idx in top_indices:
                st.write(f"- {class_names[idx]}: {result['probabilities'][idx] * 100:.2f}%")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def render_stream_tab(assets):
    st.caption("Webcam stream chay trong browser qua streamlit-webrtc.")
    webrtc_streamer(
        key="sign-language-stream",
        video_processor_factory=lambda: SignVideoProcessor(assets),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def main():
    st.set_page_config(page_title="Sign Language Test", layout="wide")
    st.title("Sign Language Test App")
    st.write("Test model bang upload video hoac webcam stream.")

    default_model = os.path.join("models", "cnn_lstm_best.keras")
    default_encoder = os.path.join("datasets", "processed", "label_encoder.pkl")

    with st.sidebar:
        st.header("Cau hinh")
        model_path = st.text_input("Model path", value=default_model)
        label_encoder_path = st.text_input("Label encoder path", value=default_encoder)

    st.session_state["model_path"] = model_path
    st.session_state["label_encoder_path"] = label_encoder_path

    missing_files = ensure_required_files()
    if missing_files:
        st.error("Thieu model landmark:")
        for path in missing_files:
            st.code(path)
        st.stop()

    if not os.path.exists(model_path):
        st.error(f"Khong tim thay model: {model_path}")
        st.stop()
    if not os.path.exists(label_encoder_path):
        st.error(f"Khong tim thay label encoder: {label_encoder_path}")
        st.stop()

    assets = load_assets(model_path, label_encoder_path)

    upload_tab, stream_tab = st.tabs(["Upload file", "Webcam stream"])

    with upload_tab:
        render_upload_tab(assets)

    with stream_tab:
        render_stream_tab(assets)


if __name__ == "__main__":
    main()
