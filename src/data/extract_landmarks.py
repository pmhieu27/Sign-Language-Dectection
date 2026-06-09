"""
Step 2: Trich xuat landmarks tu video bang MediaPipe (Multiprocessing)

Usage: python src/extract_landmarks.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATASET_ROOT, METADATA_PATH, LANDMARK_PATH, MODEL_LANDMARK_PATH, MODEL_FACE_LANDMARK_PATH,
    FACE_ANCHOR_INDICES, NUM_HAND_LANDMARKS
)


def extract_face_anchors(face_landmarks):
    """Lay mot tap moc mat nho de bo sung vi tri tay tuong doi voi mat."""
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
        output_facial_transformation_matrixes=False
    )
    return FaceLandmarker.create_from_options(options)


def build_hand_blocks(result):
    left_block = np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
    right_block = np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
    presence = np.zeros(2, dtype=np.float32)

    if not result.hand_landmarks:
        return left_block, right_block, presence

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        hand_vector = np.array(
            [[point.x, point.y, point.z] for point in hand_landmarks],
            dtype=np.float32
        ).flatten()
        handedness_list = result.handedness[idx] if idx < len(result.handedness) else []
        handedness = handedness_list[0].category_name if handedness_list else None
        handedness = (handedness or '').lower()

        if handedness == 'left':
            left_block = hand_vector
            presence[0] = 1.0
        elif handedness == 'right':
            right_block = hand_vector
            presence[1] = 1.0
        elif presence[0] == 0.0:
            left_block = hand_vector
            presence[0] = 1.0
        else:
            right_block = hand_vector
            presence[1] = 1.0

    return left_block, right_block, presence


def process_video(row):
    """Xu ly 1 video: doc frame -> detect -> luu landmark .npy"""
    try:
        video_id = row['id']
        label = row['label']
        person = row['person']
        video_path = os.path.join(DATASET_ROOT, row['video_path'])
        out_name = f"{video_id:04d}_{label}_{person}.npy"
        out_path = os.path.join(LANDMARK_PATH, out_name)

        if os.path.exists(out_path):
            return 'SKIP', row['video_name'], "Da co landmark (bo qua)"

        if not os.path.exists(video_path):
            return 'SKIP', row['video_name'], f"Khong tim thay: {video_path}"

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_LANDMARK_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        hand_landmarker = HandLandmarker.create_from_options(options)
        face_landmarker = create_face_landmarker()

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
                left_block, right_block, presence = build_hand_blocks(hand_result)
                if presence.sum() > 0:
                    vector = np.concatenate([left_block, right_block, last_face_vector, presence])
                    sequence.append(vector)

            frame_count += 1

        cap.release()
        hand_landmarker.close()
        face_landmarker.close()

        if len(sequence) < 10:
            return 'FAIL', row['video_name'], f"Qua it frame ({len(sequence)})"

        sequence = np.array(sequence, dtype=np.float32)
        np.save(out_path, sequence)
        return 'OK', row['video_name'], f"Luu {len(sequence)} frames"

    except Exception as exc:
        return 'FAIL', row.get('video_name', 'Unknown'), f"Loi: {str(exc)}"


def main():
    if not os.path.exists(MODEL_FACE_LANDMARK_PATH):
        print(f"Loi: Khong tim thay file face landmarker tai: {MODEL_FACE_LANDMARK_PATH}")
        print("Can them model mat (.task) de trich xuat feature mat.")
        return

    os.makedirs(LANDMARK_PATH, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    print(f"Tong so video can xu ly: {len(df)}")
    print("Bat dau extract landmarks...\n")

    tasks = df.to_dict('records')
    ok, skip, fail = 0, 0, 0
    skip_list = []
    fail_list = []
    max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Extracting", unit="video") as pbar:
            for future in as_completed(futures):
                status, video_name, message = future.result()

                if status == 'OK':
                    ok += 1
                elif status == 'SKIP':
                    skip += 1
                    skip_list.append(f"{video_name} - {message}")
                elif status == 'FAIL':
                    fail += 1
                    fail_list.append(f"{video_name} - {message}")

                pbar.update(1)

    print(f"\n{'=' * 50}")
    print("HOAN THANH")
    print(f"  OK    : {ok}")
    print(f"  Skip  : {skip}")
    print(f"  Fail  : {fail}")
    print(f"{'=' * 50}")

    if skip_list:
        print("\nDanh sach SKIP:")
        for item in skip_list:
            print(f"  - {item}")

    if fail_list:
        print("\nDanh sach FAIL:")
        for item in fail_list:
            print(f"  - {item}")

    print(f"\nLandmark luu tai: {LANDMARK_PATH}")


if __name__ == '__main__':
    main()
