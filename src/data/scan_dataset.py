"""
Step 1: Quét thư mục dataset -> sinh/cập nhật metadata.csv

Usage: python src/scan_dataset.py
"""

import csv
import os
from collections import Counter

from src import config


VIDEO_EXTENSIONS = {'.mov', '.mp4', '.avi'}
DATASET_ROOT = config.DATASET_ROOT
METADATA_PATH = config.METADATA_PATH
LANDMARK_PATH = config.LANDMARK_PATH
SELECTED_CLASSES = config.SELECTED_CLASSES


def load_existing_metadata():
    """Đọc metadata cũ để giữ nguyên id cho video đã tồn tại."""
    if not os.path.exists(METADATA_PATH):
        return {}, 1

    existing = {}
    max_id = 0
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['id'] = int(row['id'])
            existing[row['video_path']] = row
            max_id = max(max_id, row['id'])

    return existing, max_id + 1


def has_landmark(row):
    """Kiểm tra landmark đã tồn tại cho row metadata này chưa."""
    npy_name = f"{int(row['id']):04d}_{row['label']}_{row['person']}.npy"
    npy_path = os.path.join(LANDMARK_PATH, npy_name)
    return os.path.exists(npy_path)


def scan_dataset():
    """Quét tất cả video trong DATASET_ROOT và cập nhật metadata.csv."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    existing_metadata, next_id = load_existing_metadata()
    rows = []
    reused_with_landmark = 0
    reused_without_landmark = 0
    new_rows = 0

    for person in sorted(os.listdir(DATASET_ROOT)):
        person_path = os.path.join(DATASET_ROOT, person)
        if not os.path.isdir(person_path) or not person.startswith('person_'):
            continue

        for label in sorted(os.listdir(person_path)):
            label_path = os.path.join(person_path, label)
            if not os.path.isdir(label_path):
                continue

            if label not in SELECTED_CLASSES:
                continue

            for video_file in sorted(os.listdir(label_path)):
                ext = os.path.splitext(video_file)[1].lower()
                if ext not in VIDEO_EXTENSIONS:
                    continue

                video_path = os.path.join(person, label, video_file)
                existing_row = existing_metadata.get(video_path)

                if existing_row is not None:
                    row = {
                        'id': int(existing_row['id']),
                        'video_name': existing_row['video_name'],
                        'label': existing_row['label'],
                        'person': existing_row['person'],
                        'video_path': existing_row['video_path'],
                    }
                    rows.append(row)
                    if has_landmark(row):
                        reused_with_landmark += 1
                    else:
                        reused_without_landmark += 1
                    continue

                row = {
                    'id': next_id,
                    'video_name': video_file,
                    'label': label,
                    'person': person,
                    'video_path': video_path,
                }
                rows.append(row)
                next_id += 1
                new_rows += 1

    with open(METADATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'video_name', 'label', 'person', 'video_path'])
        writer.writeheader()
        writer.writerows(rows)

    labels = Counter(r['label'] for r in rows)
    persons = Counter(r['person'] for r in rows)

    print(f"Đã quét {len(rows)} videos -> {METADATA_PATH}")
    print(f"\nSố class: {len(labels)}")
    print(f"Số người: {len(persons)}")
    for person_name, count in sorted(persons.items()):
        print(f"  {person_name}: {count} videos")

    print(f"\nSố videos trên mỗi class:")
    for label_name, count in sorted(labels.items()):
        print(f"  {label_name:15s}: {count}")

    print(f"\nTái sử dụng metadata:")
    print(f"  Tái sử dụng với landmark   : {reused_with_landmark}")
    print(f"  Tái sử dụng không với landmark: {reused_without_landmark}")
    print(f"  Dòng mới                    : {new_rows}")


if __name__ == '__main__':
    scan_dataset()
