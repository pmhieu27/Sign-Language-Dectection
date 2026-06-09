# Sign Language Detection

Nhận diện ngôn ngữ ký hiệu Việt Nam (10 class) sử dụng MediaPipe landmarks + CNN1D-LSTM.

## Cấu trúc dự án

```text
sign_language/
├── src/                        # Package Python chính
│   ├── config.py               # Cấu hình paths, constants, classes
│   ├── data/                   # Pipeline xử lý dữ liệu
│   │   ├── scan_dataset.py     #   Quét dataset → metadata.csv
│   │   ├── extract_landmarks.py#   Trích xuất landmarks từ video
│   │   ├── preprocess.py       #   Normalize, pad, split train/val/test
│   │   └── augmentation.py     #   Data augmentation cho train set
│   ├── training/               # Huấn luyện & đánh giá
│   │   ├── models.py           #   Kiến trúc LSTM, CNN1D+LSTM
│   │   ├── train_lstm.py       #   Train LSTM
│   │   ├── train_cnn_lstm.py   #   Train CNN1D+LSTM
│   │   ├── train_loso.py       #   LOSO cross-validation
│   │   ├── train_final.py      #   Train final model (merge val)
│   │   └── evaluate.py         #   Đánh giá trên test set
│   ├── inference/              # Suy luận real-time
│   │   ├── pipeline.py         #   Shared runtime helpers
│   │   └── stream.py           #   Webcam stream
│   └── apps/                   # Ứng dụng
│       └── streamlit_app.py    #   Streamlit web app
│
├── scripts/                    # CLI entry points
│   ├── prepare_data.py         #   Pipeline chuẩn bị data
│   ├── train.py                #   CLI chọn model train
│   └── evaluate.py             #   CLI evaluate
│
├── datasets/                   # Dữ liệu (không push git)
│   ├── dataset/                #   Video gốc
│   ├── landmarks/              #   Landmarks .npy
│   ├── metadata/               #   metadata.csv, labels.csv
│   └── processed/              #   Train/val/test splits
│
└── models/                     # Model files
    ├── mediapipe/              #   MediaPipe tasks (.task)
    ├── *.keras                 #   Model weights (không push git)
    └── *.json                  #   Training history
```

## Cách sử dụng

### 1. Chuẩn bị dữ liệu

```bash
# Chạy toàn bộ pipeline
python scripts/prepare_data.py

# Hoặc chạy từng bước
python scripts/prepare_data.py --step scan
python scripts/prepare_data.py --step extract
python scripts/prepare_data.py --step preprocess
python scripts/prepare_data.py --step augment
```

### 2. Huấn luyện

```bash
python scripts/train.py cnn_lstm               # CNN1D+LSTM (khuyến nghị)
python scripts/train.py lstm                    # LSTM
python scripts/train.py loso                    # LOSO cross-validation
python scripts/train.py final                   # Final model

# Với options
python scripts/train.py cnn_lstm --epochs 200 --batch_size 16
```

### 3. Đánh giá

```bash
python scripts/evaluate.py
```

### 4. stream

```bash
# Webcam stream
python -m src.inference.stream --model_path models/cnn_lstm_best.keras

# Streamlit app
streamlit run src/apps/streamlit_app.py
```

## 10 Classes

| # | Ký hiệu | Ý nghĩa |
|---|---------|---------|
| 1 | Xin chào | Chào hỏi |
| 2 | Tạm biệt | Chia tay |
| 3 | Cảm ơn | Lịch sự |
| 4 | Xin lỗi | Lịch sự |
| 5 | Có | Trả lời |
| 6 | Không | Trả lời |
| 7 | Ăn | Nhu cầu |
| 8 | Uống | Nhu cầu |
| 9 | Tôi | Đại từ |
| 10 | Hạnh phúc | Cảm xúc |
