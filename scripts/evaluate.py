"""
CLI đánh giá model trên test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --data_dir /path/to/processed --model_path /path/to/model.keras
"""

import sys

# Thiết lập encoding UTF-8 cho stdout/stderr để tránh lỗi hiển thị ký tự Unicode trên Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def main():
    from src.training.evaluate_final import main as evaluate_main
    evaluate_main()


if __name__ == '__main__':
    main()
