# src/visualize_landmark.py
import cv2
import mediapipe as mp
import os

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

# Chọn 1 video bất kỳ có landmark tốt
VIDEO_PATH = r'C:\AI\Sign-Language-Dectection\datasets\Videos\Huyền\Xin chào\Huyen_252.MOV'
OUTPUT_DIR = r'C:\AI\Sign-Language-Dectection'
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
print("Mở video được không:", cap.isOpened())
print("Tổng frame:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
saved = 0

while cap.isOpened() and saved < 3:
    ret, frame = cap.read()
    print("Đọc frame:", ret, "| saved:", saved)
    if not ret:
        break

    if saved == 0:
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'before_landmark.jpg'), frame)
        print("Đã lưu before_landmark.jpg")

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    print("Detect tay:", result.multi_hand_landmarks is not None)
    if result.multi_hand_landmarks:
        annotated = frame.copy()
        for hand_lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2)
            )
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'after_landmark_{saved}.jpg'), annotated)
        print(f"Đã lưu after_landmark_{saved}.jpg")
        saved += 1

cap.release()
hands.close()
print(f"Đã lưu ảnh vào: {OUTPUT_DIR}")
print("- before_landmark.jpg  → ảnh gốc (trước)")
print("- after_landmark_0.jpg → ảnh có landmark (sau)")