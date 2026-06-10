import os
import pickle
import tempfile
import time
import textwrap
import requests
from collections import deque
from dataclasses import dataclass

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow import keras

from src.config import (
    MODEL_FACE_LANDMARK_PATH,
    MODEL_LANDMARK_PATH,
    SEQ_LEN,
    FACE_ANCHOR_INDICES,
    NUM_HAND_LANDMARKS,
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


METERED_API_URL = "https://signlanguagerecog.metered.live/api/v1/turn/credentials?apiKey=343310862d18295c1bac21748c7ff2429ee2"


@st.cache_data(ttl=3600)
def get_ice_servers():
    """Fetch TURN/STUN credentials from Metered.ca API (cached 1 hour)."""
    try:
        resp = requests.get(METERED_API_URL, timeout=10)
        resp.raise_for_status()
        ice_servers = resp.json()
        if ice_servers:
            return ice_servers
    except Exception as e:
        print(f"Warning: Failed to fetch TURN credentials: {e}")
    # Fallback to free STUN only
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


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

        # UI Metrics
        self.top_k_predictions = []
        self.frame_times = deque(maxlen=30)
        self.fps = 0.0
        self.latency = 0.0
        self.history = []
        self.last_stable_label = None

    def recv(self, frame):
        t_start = time.time()

        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(self.frame_index * (1000 / 30.0))
        self.frame_index += 1

        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        height, width, _ = image.shape
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
                t_pred_start = time.time()
                result = predict_from_sequence(np.array(self.sequence_buffer, dtype=np.float32), self.assets)
                self.latency = (time.time() - t_pred_start) * 1000
                if result is not None:
                    self.predicted_label = result["label"]
                    self.confidence = result["confidence"]

                    # Extract Top 3
                    class_names = list(self.assets.label_encoder.classes_)
                    probs = result["probabilities"]
                    top_indices = np.argsort(probs)[::-1][:3]
                    self.top_k_predictions = [(class_names[idx], float(probs[idx])) for idx in top_indices]

                    # Append to History if confidence > 0.6
                    if self.confidence >= 0.6:
                        if self.predicted_label != self.last_stable_label:
                            self.last_stable_label = self.predicted_label
                            self.history.append((self.predicted_label, self.confidence))
                            if len(self.history) > 10:
                                self.history.pop(0)

        # FPS Calculation
        self.frame_times.append(time.time())
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        else:
            self.fps = 30.0

        # Return clean image with absolutely no drawings
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
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h3 style="margin-top:0; color:#818CF8; font-size:18px; font-weight:700;">CHẾ ĐỘ NHẬN DIỆN QUA VIDEO TẢI LÊN</h3>
        <p style="color:#94A3B8; font-size:14px; margin-bottom: 10px;">Tải lên file video từ máy tính để thực hiện dịch ngôn ngữ ký hiệu.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload video (.mov, .mp4, .avi)",
        type=["mov", "mp4", "avi"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.session_state.upload_result = None
        return

    st.video(uploaded_file)

    if st.button("Chạy nhận diện video", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Đang trích xuất landmark và dự đoán..."):
                sequence = extract_sequence_from_video(tmp_path)
                if sequence is None or len(sequence) < 10:
                    st.error("Không đủ frame landmark hợp lệ để dự đoán.")
                    st.session_state.upload_result = None
                    return
                result = predict_from_sequence(sequence, assets)
                if result is None:
                    st.error("Không thể dự đoán trên video này.")
                    st.session_state.upload_result = None
                    return

            st.session_state.upload_result = result
            st.session_state.upload_seq_len = len(sequence)
            st.rerun()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def main():
    st.set_page_config(page_title="Nhận diện Ngôn ngữ Ký hiệu", layout="wide")

    # Custom CSS Styling (Dark Theme & Button Highlight & Black Text Buttons)
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        .stApp {
            background-color: #0B0F19 !important;
            color: #F8FAFC !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1.5rem !important;
            max-width: 100% !important;
        }
        
        .dashboard-card {
            background-color: #131B2E;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        
        .dashboard-title {
            color: #94A3B8;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }
        
        .main-prediction-text {
            font-size: 42px;
            font-weight: 800;
            color: #FFFFFF;
            margin: 8px 0;
            line-height: 1.1;
        }
        
        /* Make Radio Option Text Bold and White */
        div[data-testid="stRadio"] label p {
            color: #FFFFFF !important;
            font-weight: bold !important;
            font-size: 15px !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 20px !important;
            margin-bottom: 15px !important;
        }
        
        /* Style secondary buttons (like Dừng, Xóa lịch sử) to have bold black text */
        div[data-testid="stElementContainer"] button:not([kind="primary"]) p,
        div[data-testid="stElementContainer"] button:not([kind="primary"]) span,
        div[data-testid="stElementContainer"] button:not([kind="primary"]),
        div[data-testid="stBlock"] button:not([kind="primary"]) p,
        div[data-testid="stBlock"] button:not([kind="primary"]) span,
        div[data-testid="stBlock"] button:not([kind="primary"]) {
            color: #000000 !important;
            font-weight: 800 !important;
            background-color: #E2E8F0 !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
        }
        
        div[data-testid="stElementContainer"] button:not([kind="primary"]):hover p,
        div[data-testid="stElementContainer"] button:not([kind="primary"]):hover span,
        div[data-testid="stElementContainer"] button:not([kind="primary"]):hover,
        div[data-testid="stBlock"] button:not([kind="primary"]):hover p,
        div[data-testid="stBlock"] button:not([kind="primary"]) :hover span,
        div[data-testid="stBlock"] button:not([kind="primary"]):hover {
            color: #000000 !important;
            background-color: #CBD5E1 !important;
            border-color: #94A3B8 !important;
        }
        
        /* Specific green highlight for primary buttons */
        div[data-testid="stElementContainer"] button[kind="primary"] p,
        div[data-testid="stElementContainer"] button[kind="primary"] span,
        div[data-testid="stElementContainer"] button[kind="primary"],
        div[data-testid="stBlock"] button[kind="primary"] p,
        div[data-testid="stBlock"] button[kind="primary"] span,
        div[data-testid="stBlock"] button[kind="primary"] {
            background-color: #10B981 !important;
            color: #FFFFFF !important;
            border: 1px solid #10B981 !important;
            font-weight: 800 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
        }
        div[data-testid="stElementContainer"] button[kind="primary"]:hover p,
        div[data-testid="stElementContainer"] button[kind="primary"]:hover span,
        div[data-testid="stElementContainer"] button[kind="primary"]:hover,
        div[data-testid="stBlock"] button[kind="primary"]:hover p,
        div[data-testid="stBlock"] button[kind="primary"]:hover span,
        div[data-testid="stBlock"] button[kind="primary"]:hover {
            background-color: #059669 !important;
            border-color: #059669 !important;
            color: #FFFFFF !important;
        }
        
        /* Hide webrtc device selection list */
        div[data-testid="stSelectbox"] {
            display: none !important;
        }
        
        .history-badge {
            background-color: #1E293B;
            border: 1px solid #334155;
            color: #F8FAFC;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-right: 8px;
            margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # OS Window Title Bar
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#111827; padding:10px 20px; border-radius:8px 8px 0 0; border-bottom:1px solid #1E293B; margin-bottom:20px;">
        <div style="display:flex; gap:8px;">
            <div style="width:12px; height:12px; background:#EF4444; border-radius:50%;"></div>
            <div style="width:12px; height:12px; background:#F59E0B; border-radius:50%;"></div>
            <div style="width:12px; height:12px; background:#10B981; border-radius:50%;"></div>
        </div>
        <div style="color:#F8FAFC; font-size:14px; font-weight:700; display:flex; align-items:center; gap:8px;">
            Nhận diện ngôn ngữ ký hiệu
        </div>
        <div style="color:#64748B; font-size:12px; font-weight:700;">IS54A - Trí tuệ nhân tạo</div>
    </div>
    """, unsafe_allow_html=True)

    default_model = os.path.join("models", "cnn_lstm_final.keras")
    default_encoder = os.path.join("datasets", "processed", "label_encoder.pkl")

    missing_files = ensure_required_files()
    if missing_files:
        st.error("Thiếu model landmark:")
        for path in missing_files:
            st.code(path)
        st.stop()

    if not os.path.exists(default_model):
        st.error(f"Không tìm thấy model: {default_model}")
        st.stop()
    if not os.path.exists(default_encoder):
        st.error(f"Không tìm thấy label encoder: {default_encoder}")
        st.stop()

    assets = load_assets(default_model, default_encoder)

    # Horizontal Tab Selection (Radio styled as segmented tabs)
    app_mode = st.radio(
        "Nguồn dữ liệu:",
        ["Webcam", "Upload Video"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Session State setup
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = True
    if "upload_result" not in st.session_state:
        st.session_state.upload_result = None
    if "upload_seq_len" not in st.session_state:
        st.session_state.upload_seq_len = 0
    if "prev_app_mode" not in st.session_state:
        st.session_state.prev_app_mode = app_mode

    # Clear results on mode switch
    if st.session_state.prev_app_mode != app_mode:
        st.session_state.prev_app_mode = app_mode
        st.session_state.upload_result = None
        st.session_state.webcam_active = True

    # Main dashboard columns (Left Column 8/12 | Right Column 4/12)
    col_left, col_right = st.columns([8, 4])

    with col_left:
        if app_mode == "Webcam":
            ctx = None
            if st.session_state.webcam_active:
                ctx = webrtc_streamer(
                    key="sign-language-stream",
                    video_processor_factory=lambda: SignVideoProcessor(assets),
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration={
                        "iceServers": get_ice_servers()
                    },
                    async_processing=True,
                )
            else:
                st.markdown("""
                <div style="background:#131B2E; border:1px solid #1E293B; border-radius:12px; height:340px; display:flex; flex-direction:column; justify-content:center; align-items:center; color:#94A3B8;">
                    <div style="font-size:16px; font-weight:700;">Webcam feed đang tắt</div>
                    <div style="font-size:13px; color:#64748B; margin-top:6px;">Nhấn nút Bật webcam để bắt đầu nhận diện</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            
            # Control Buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("Bật webcam", type="primary", use_container_width=True):
                    st.session_state.webcam_active = True
                    st.rerun()
            with col_btn2:
                if st.button("Dừng", use_container_width=True):
                    st.session_state.webcam_active = False
                    st.rerun()
            with col_btn3:
                if st.button("Xóa lịch sử", use_container_width=True):
                    if ctx and ctx.video_processor:
                        ctx.video_processor.history = []
                    st.rerun()
        else:
            # Upload Video Mode
            render_upload_tab(assets)

    with col_right:
        pred_placeholder = st.empty()
        top_placeholder = st.empty()
        status_placeholder = st.empty()
        history_placeholder = st.empty()

        # Render placeholders initially
        pred_placeholder.markdown("""
<div class="dashboard-card">
    <div class="dashboard-title">KẾT QUẢ NHẬN DIỆN</div>
    <div class="main-prediction-text" style="color: #475569;">...</div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px;">
        <span style="color:#94A3B8; font-size:14px;">Độ tin cậy</span>
        <span style="color:#475569; font-weight:700; font-size:16px;">0.0%</span>
    </div>
    <div style="height:8px; background:#1E293B; border-radius:4px; overflow:hidden; margin-top:8px;">
        <div style="width:0%; height:100%; background:#10B981; border-radius:4px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        top_placeholder.markdown("""
<div class="dashboard-card">
    <div class="dashboard-title">TOP 3 DỰ ĐOÁN</div>
    <div style="color:#64748B; font-size:14px; font-style:italic; margin-top:8px;">Chưa có dữ liệu dự đoán...</div>
</div>
""", unsafe_allow_html=True)

        status_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">TRẠNG THÁI MODEL</div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Trạng thái</span>
        <span style="color:#EF4444; font-weight:700;">DỪNG</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Tốc độ xử lý (FPS)</span>
        <span style="color:#475569; font-weight:700;">0.0 FPS</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Độ trễ (Latency)</span>
        <span style="color:#475569; font-weight:700;">~0ms</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:15px;">
        <span style="color:#94A3B8;">Bộ đệm (Buffer)</span>
        <span style="color:#475569; font-weight:700;">0 / {SEQ_LEN}</span>
    </div>
</div>
""", unsafe_allow_html=True)

        history_placeholder.markdown("""
<div class="dashboard-card">
    <div class="dashboard-title">LỊCH SỬ PHIÊN NÀY</div>
    <span style="color:#64748B; font-style:italic; font-size:14px;">Chưa ghi nhận ký hiệu nào...</span>
</div>
""", unsafe_allow_html=True)

    # ── REALTIME WEB-SOCKET UPDATE LOOP (Webcam mode) ──
    if app_mode == "Webcam":
        if ctx and ctx.video_processor:
            while ctx.state.playing:
                processor = ctx.video_processor
                
                predicted_label = processor.predicted_label
                confidence = processor.confidence
                top_k_predictions = processor.top_k_predictions
                fps = processor.fps
                latency = processor.latency
                buffer_len = len(processor.sequence_buffer)
                history = processor.history
                
                is_below = confidence < 0.6 or predicted_label == "..."
                label_color = "#475569" if is_below else "#FFFFFF"
                label_text = "..." if is_below else predicted_label
                
                pred_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">KẾT QUẢ NHẬN DIỆN</div>
    <div class="main-prediction-text" style="color: {label_color};">{label_text}</div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px;">
        <span style="color:#94A3B8; font-size:14px;">Độ tin cậy</span>
        <span style="color:#10B981; font-weight:700; font-size:16px;">{confidence * 100:.1f}%</span>
    </div>
    <div style="height:8px; background:#1E293B; border-radius:4px; overflow:hidden; margin-top:8px;">
        <div style="width:{confidence * 100}%; height:100%; background:#10B981; border-radius:4px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)
                
                top_3_html = ""
                if len(top_k_predictions) > 0:
                    for label, prob in top_k_predictions:
                        top_3_html += f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
    <span style="font-size:15px; font-weight:600; color:#F1F5F9; min-width:80px;">{label}</span>
    <div style="flex-grow:1; margin:0 16px; height:6px; background:#1E293B; border-radius:3px; overflow:hidden;">
        <div style="width:{prob * 100}%; height:100%; background:#6366F1; border-radius:3px;"></div>
    </div>
    <span style="font-size:14px; color:#94A3B8; font-weight:700; min-width:40px; text-align:right;">{prob * 100:.0f}%</span>
</div>
"""
                else:
                    top_3_html = '<div style="color:#64748B; font-size:14px; font-style:italic; margin-top:8px;">Chưa có dữ liệu dự đoán...</div>'
                    
                top_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">TOP 3 DỰ ĐOÁN</div>
    {top_3_html}
</div>
""", unsafe_allow_html=True)
                
                buf_color = "#10B981" if buffer_len >= SEQ_LEN else "#F59E0B"
                status_text = "ĐANG CHẠY" if ctx.state.playing else "DỪNG"
                status_color = "#10B981" if ctx.state.playing else "#EF4444"
                
                status_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">TRẠNG THÁI MODEL</div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Trạng thái</span>
        <span style="color:{status_color}; font-weight:700;">{status_text}</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Tốc độ xử lý (FPS)</span>
        <span style="color:#FFFFFF; font-weight:700;">{fps:.1f} FPS</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Độ trễ (Latency)</span>
        <span style="color:#F59E0B; font-weight:700;">~{latency:.0f}ms</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:15px;">
        <span style="color:#94A3B8;">Bộ đệm (Buffer)</span>
        <span style="color:{buf_color}; font-weight:700;">{buffer_len} / {SEQ_LEN}</span>
    </div>
</div>
""", unsafe_allow_html=True)
                
                history_html = '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
                if len(history) > 0:
                    for lbl, conf in reversed(history):
                        history_html += f"""
<div class="history-badge">
    <span style="font-weight:700; color:#FFFFFF;">{lbl}</span>
    <span style="font-size:11px; color:#10B981; font-weight:700;">{conf * 100:.0f}%</span>
</div>
"""
                else:
                    history_html += '<span style="color:#64748B; font-style:italic; font-size:14px;">Chưa ghi nhận ký hiệu nào...</span>'
                history_html += '</div>'
                
                history_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">LỊCH SỬ PHIÊN NÀY</div>
    {history_html}
</div>
""", unsafe_allow_html=True)
                pred_changed = (predicted_label != st.session_state.get("last_ui_label", ""))
                
                if pred_changed:
                    st.session_state.last_ui_label = predicted_label
                    st.rerun()
                else:
                    time.sleep(0.1)
    else:
        # Upload Video Mode predictions
        if st.session_state.upload_result is not None:
            res = st.session_state.upload_result
            predicted_label = res["label"]
            confidence = res["confidence"]
            
            # Top 3
            class_names = list(assets.label_encoder.classes_)
            probs = res["probabilities"]
            top_indices = np.argsort(probs)[::-1][:3]
            top_k = [(class_names[idx], float(probs[idx])) for idx in top_indices]
            
            # 1. Prediction Card
            pred_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">KẾT QUẢ NHẬN DIỆN</div>
    <div class="main-prediction-text" style="color: #FFFFFF;">{predicted_label}</div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px;">
        <span style="color:#94A3B8; font-size:14px;">Độ tin cậy</span>
        <span style="color:#10B981; font-weight:700; font-size:16px;">{confidence * 100:.1f}%</span>
    </div>
    <div style="height:8px; background:#1E293B; border-radius:4px; overflow:hidden; margin-top:8px;">
        <div style="width:{confidence * 100}%; height:100%; background:#10B981; border-radius:4px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # 2. Top 3 Card
            top_3_html = ""
            for label, prob in top_k:
                top_3_html += f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
    <span style="font-size:15px; font-weight:600; color:#F1F5F9; min-width:80px;">{label}</span>
    <div style="flex-grow:1; margin:0 16px; height:6px; background:#1E293B; border-radius:3px; overflow:hidden;">
        <div style="width:{prob * 100}%; height:100%; background:#6366F1; border-radius:3px;"></div>
    </div>
    <span style="font-size:14px; color:#94A3B8; font-weight:700; min-width:40px; text-align:right;">{prob * 100:.0f}%</span>
</div>
"""
            top_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">TOP 3 DỰ ĐOÁN</div>
    {top_3_html}
</div>
""", unsafe_allow_html=True)
            
            # 3. Model Status Card
            seq_len_show = st.session_state.get("upload_seq_len", SEQ_LEN)
            status_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">TRẠNG THÁI MODEL</div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Trạng thái</span>
        <span style="color:#10B981; font-weight:700;">HOÀN THÀNH</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Tốc độ xử lý (FPS)</span>
        <span style="color:#FFFFFF; font-weight:700;">N/A</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:15px;">
        <span style="color:#94A3B8;">Độ trễ (Latency)</span>
        <span style="color:#FFFFFF; font-weight:700;">N/A</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:15px;">
        <span style="color:#94A3B8;">Bộ đệm (Buffer)</span>
        <span style="color:#10B981; font-weight:700;">{seq_len_show} / {SEQ_LEN}</span>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # 4. History Card
            history_placeholder.markdown(f"""
<div class="dashboard-card">
    <div class="dashboard-title">LỊCH SỬ PHIÊN NÀY</div>
    <div class="history-badge">
        <span style="font-weight:700; color:#FFFFFF;">{predicted_label}</span>
        <span style="font-size:11px; color:#10B981; font-weight:700;">{confidence * 100:.0f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
