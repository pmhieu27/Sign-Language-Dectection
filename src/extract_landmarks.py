import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

VIDEO_ROOT = r'C:\AI\Sign-Language-Dectection\datasets\Videos'
OUTPUT_DIR = r'C:\AI\Sign-Language-Dectection\datasets\landmarks'
CSV_PATH   = r'C:\AI\Sign-Language-Dectection\datasets\metadata\dataset.csv'

PERSON_MAP = {
    'huyen': 'Huyền',
    'kieu' : 'Kiều',
    'kiều' : 'Kiều',
    'thu'  : 'Thu',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df['video_name'] = df['video_name'].str.strip()
df['person']     = df['person'].str.strip().str.lower()
df['label']      = df['label'].str.strip()

ok, skip, fail = 0, 0, 0
skip_list = []
fail_list = []

print(f"Tổng số video cần xử lý: {len(df)}")
print("Bắt đầu extract landmarks...\n")

for _, row in df.iterrows():
    video_name = row['video_name']
    label      = row['label']
    person_key = row['person']
    person_dir = PERSON_MAP.get(person_key, person_key.capitalize())
# Bỏ qua nếu đã extract rồi
    out_name = f"{row['id']:04d}_{label}_{person_key}.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if os.path.exists(out_path):
        ok += 1
        continue
    person_path = os.path.join(VIDEO_ROOT, person_dir)
    if not os.path.exists(person_path):
        msg = f"Không tìm thấy folder người: {person_dir}"
        print(f"[SKIP] {msg}")
        skip_list.append(f"{video_name} — {msg}")
        skip += 1
        continue

    # Tìm thư mục nhãn (so sánh không phân biệt hoa thường)
    label_folder = None
    for folder in os.listdir(person_path):
        if folder.strip().lower() == label.strip().lower():
            label_folder = folder
            break

    if label_folder is None:
        msg = f"Không tìm thấy folder nhãn '{label}' trong {person_dir}"
        print(f"[SKIP] {msg}")
        skip_list.append(f"{video_name} — {msg}")
        skip += 1
        continue

    video_path = os.path.join(person_path, label_folder, video_name)
    if not os.path.exists(video_path):
        msg = f"Không tìm thấy file video: {video_path}"
        print(f"[SKIP] {msg}")
        skip_list.append(f"{video_name} — {msg}")
        skip += 1
        continue

    # Đọc video và extract landmark
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            vector = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
            sequence.append(vector)
    cap.release()

    if len(sequence) < 10:
        msg = f"Quá ít frame detect được ({len(sequence)} frame)"
        print(f"[FAIL] {video_name} — {msg}")
        fail_list.append(f"{video_name} — {msg}")
        fail += 1
        continue

    sequence = np.array(sequence)

    # Lưu .npy — tên file: id_label_person
    out_name = f"{row['id']:04d}_{label}_{person_key}.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.save(out_path, sequence)
    ok += 1

    if ok % 20 == 0:
        print(f"  [{ok}/{len(df)}] Đã xử lý {ok} video...")

hands.close()

print(f"\n{'='*50}")
print(f"HOÀN THÀNH")
print(f"  OK    : {ok}")
print(f"  Skip  : {skip}")
print(f"  Fail  : {fail}")
print(f"{'='*50}")

if skip_list:
    print("\nDanh sách SKIP:")
    for s in skip_list:
        print(f"  - {s}")

if fail_list:
    print("\nDanh sách FAIL:")
    for f in fail_list:
        print(f"  - {f}")

print(f"\nLandmark lưu tại: {OUTPUT_DIR}")