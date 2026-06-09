"""
CLI huấn luyện model.

Usage:
    python scripts/train.py lstm                    # Train LSTM
    python scripts/train.py cnn_lstm                # Train CNN+LSTM
    python scripts/train.py loso                    # LOSO cross-validation
    python scripts/train.py final                   # Train final model

    Truyền thêm arguments cho script gốc:
    python scripts/train.py cnn_lstm --epochs 200 --batch_size 16
"""

import sys


MODELS = {
    'loso': 'src.training.train_loso',
    'final_best': 'src.training.train_best_final',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python scripts/train.py <model> [options]")
        print(f"\nAvailable models: {', '.join(MODELS.keys())}")
        print("\nOptions sẽ được truyền thẳng cho script training tương ứng.")
        print("\nVí dụ:")
        print("  python scripts/train.py cnn_lstm --epochs 200")
        print("  python scripts/train.py loso --data_dir /path/to/data")
        sys.exit(0)

    model_name = sys.argv[1]
    if model_name not in MODELS:
        print(f"❌ Model '{model_name}' không hợp lệ.")
        print(f"   Chọn một trong: {', '.join(MODELS.keys())}")
        sys.exit(1)

    # Loại bỏ tên model khỏi sys.argv để script gốc parse đúng
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    module_path = MODELS[model_name]
    print(f"🚀 Training model: {model_name}")
    print(f"   Module: {module_path}")
    print("=" * 60)

    import importlib
    module = importlib.import_module(module_path)
    module.main()


if __name__ == '__main__':
    main()
