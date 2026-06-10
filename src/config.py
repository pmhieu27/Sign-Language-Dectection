import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== Paths =====
DATASET_ROOT = os.path.join(BASE_DIR, os.getenv('DATASET_ROOT', 'datasets\dataset'))
METADATA_PATH = os.path.join(BASE_DIR, os.getenv('METADATA_PATH', 'datasets\metadata\metadata.csv'))
LANDMARK_PATH = os.path.join(BASE_DIR, os.getenv('LANDMARK_PATH', 'datasets\landmarks'))
PROCESSED_PATH = os.path.join(BASE_DIR, os.getenv('PROCESSED_PATH', 'datasets\processed'))
MODEL_LANDMARK_PATH = os.path.join(BASE_DIR, os.getenv('MODEL_LANDMARK_PATH', 'models\mediapipe\hand_landmarker.task'))
MODEL_FACE_LANDMARK_PATH = os.path.join(BASE_DIR, os.getenv('MODEL_FACE_LANDMARK_PATH', 'models\mediapipe\face_landmarker.task'))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')

# ===== Constants =====
SEQ_LEN = 30

# ===== Landmark feature layout =====
NUM_HANDS = 2
NUM_HAND_LANDMARKS = 21
FACE_ANCHOR_INDICES = [1, 33, 263, 61, 291, 152]  # nose, eyes, mouth corners, chin
NUM_FACE_LANDMARKS = len(FACE_ANCHOR_INDICES)

HAND_COORD_FEATURES = NUM_HANDS * NUM_HAND_LANDMARKS * 3
FACE_COORD_FEATURES = NUM_FACE_LANDMARKS * 3
COORD_FEATURES = HAND_COORD_FEATURES + FACE_COORD_FEATURES
PRESENCE_FEATURES = NUM_HANDS
RAW_FEATURES = COORD_FEATURES + PRESENCE_FEATURES
INTER_HAND_FEATURES = 13
NUM_FEATURES = RAW_FEATURES + INTER_HAND_FEATURES

# ===== Subject-independent split =====
TRAIN_PERSONS = ['person_01', 'person_02', 'person_03']
VAL_PERSONS   = ['person_07']
TEST_PERSONS  = ['person_08']

# ===== 10 class được chọn =====
SELECTED_CLASSES = [
    'Xin chào',     # Chào hỏi
    'Tạm biệt',    # Chia tay
    'Cảm ơn',      # Lịch sự
    'Xin lỗi',     # Lịch sự
    'Có',           # Trả lời
    'Không',        # Trả lời
    'Ăn',           # Nhu cầu
    'Uống',         # Nhu cầu
    'Tôi',          # Đại từ
    'Hạnh phúc',    # Cảm xúc
    "Bố",
    "Hiểu",
    "Học",
    "Thích",
    "Bạn bè"
]
