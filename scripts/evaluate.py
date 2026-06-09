"""
CLI đánh giá model trên test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --data_dir /path/to/processed --model_path /path/to/model.keras
"""

import sys


def main():
    from src.training.evaluate_final_best import main as evaluate_main
    evaluate_main()


if __name__ == '__main__':
    main()
