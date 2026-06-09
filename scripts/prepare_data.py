"""
Pipeline chuẩn bị dữ liệu — chạy tuần tự các bước.

Usage:
    python scripts/prepare_data.py                  # Chạy tất cả
    python scripts/prepare_data.py --step scan       # Chỉ scan
    python scripts/prepare_data.py --step extract    # Chỉ extract landmarks
    python scripts/prepare_data.py --step preprocess # Chỉ preprocess
    python scripts/prepare_data.py --step augment    # Chỉ augment
"""

import argparse
import sys

STEPS = ['scan', 'extract', 'preprocess', 'augment']


def run_scan():
    from src.data.scan_dataset import scan_dataset
    print("\n" + "=" * 60)
    print("  STEP 1: Scan Dataset → metadata.csv")
    print("=" * 60)
    scan_dataset()


def run_extract():
    from src.data.extract_landmarks import main
    print("\n" + "=" * 60)
    print("  STEP 2: Extract Landmarks")
    print("=" * 60)
    main()


def run_preprocess():
    from src.data.preprocess import preprocess
    print("\n" + "=" * 60)
    print("  STEP 3: Preprocess → train/val/test splits")
    print("=" * 60)
    preprocess()


def run_augment():
    from src.data.augmentation import augment_train
    print("\n" + "=" * 60)
    print("  STEP 4: Augment train set")
    print("=" * 60)
    augment_train()


STEP_MAP = {
    'scan': run_scan,
    'extract': run_extract,
    'preprocess': run_preprocess,
    'augment': run_augment,
}


def main():
    parser = argparse.ArgumentParser(description='Prepare data pipeline')
    parser.add_argument(
        '--step', type=str, default=None, choices=STEPS,
        help='Chạy một bước cụ thể. Bỏ trống để chạy tất cả.'
    )
    args = parser.parse_args()

    if args.step:
        STEP_MAP[args.step]()
    else:
        for step_name in STEPS:
            STEP_MAP[step_name]()

    print("\n✅ Hoàn tất!")


if __name__ == '__main__':
    main()
