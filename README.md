# Sign Language Recognition

Nhận diện từ vựng ngôn ngữ ký hiệu từ video bằng pipeline:

- trích xuất `MediaPipe landmarks`
- chuẩn hóa chuỗi đặc trưng
- huấn luyện `CNN1D + LSTM`
- đánh giá theo `LOSO` và test set riêng

## 1. Cấu trúc chính

```text
sign_language/
├── datasets/
│   ├── dataset/              # Video gốc theo person/label
│   ├── landmarks/            # File .npy sau khi extract landmarks
│   ├── metadata/             # metadata.csv, labels.csv
│   └── processed/            # X_train, X_val, X_test, X_all...
├── models/
│   ├── mediapipe/            # hand_landmarker.task, face_landmarker.task
│   └── *.keras / *.json      # model và báo cáo huấn luyện
├── scripts/
│   ├── prepare_data.py       # chạy pipeline chuẩn bị dữ liệu
│   ├── train.py              # train LOSO / train model cuối
│   └── evaluate.py           # đánh giá model cuối
└── src/
    ├── data/
    ├── training/
    ├── inference/
    └── apps/
```

## 2. Chuẩn bị dữ liệu

Đặt video theo cấu trúc:

```text
datasets/dataset/
├── person_01/
│   ├── Xin chào/
│   ├── Cảm ơn/
│   └── ...
├── person_02/
└── ...
```

Mỗi thư mục `person_xx` chứa các thư mục nhãn, mỗi nhãn chứa các file video `.mov`, `.mp4` hoặc `.avi`.

## 3. Cài đặt

Tạo môi trường và cài thư viện theo cách bạn đang dùng cho project. Nếu đã có `.venv` thì chỉ cần kích hoạt môi trường trước khi chạy.

Ví dụ trên Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Chuẩn bị dữ liệu huấn luyện

Chạy toàn bộ pipeline:

```bash
python scripts/prepare_data.py
```

Hoặc chạy từng bước:

```bash
python scripts/prepare_data.py --step scan
python scripts/prepare_data.py --step extract
python scripts/prepare_data.py --step preprocess
python scripts/prepare_data.py --step augment
```

Ý nghĩa từng bước:

- `scan`: quét thư mục video và sinh `datasets/metadata/metadata.csv`
- `extract`: trích xuất landmarks tay và mặt từ video
- `preprocess`: chuẩn hóa đặc trưng, pad/sample về cùng độ dài, tạo split
- `augment`: tạo file augment cho các thử nghiệm dữ liệu

## 5. Huấn luyện mô hình

### 5.1. Chạy LOSO để tuning

```bash
python scripts/train.py loso
```

Chạy với seed cụ thể:

```bash
python scripts/train.py loso --seed 42
python scripts/train.py loso --seed 123
python scripts/train.py loso --seed 3407
```

Bạn nên dùng `LOSO` để:

- so sánh augmentation
- tuning learning rate
- tuning batch size
- tuning kiến trúc

### 5.2. Huấn luyện model cuối

Model cuối hiện dùng cấu hình đã tune:

- augmentation: `speed 0.9`, `speed 1.1`, `jitter 0.02`, `rotation 3`
- model: `Conv1D(64) -> Conv1D(128) -> LSTM(64) -> Dense(64)`
- dropout: `0.3 / 0.4 / 0.2`
- optimizer: `Adam`
- learning rate: `7e-4`
- batch size: `32`

Chạy model cuối:

```bash
python scripts/train.py final_best
```

Ví dụ:

```bash
python scripts/train.py final_best --epochs 150 --seed 42
python scripts/train.py final_best --epochs 150 --seed 123
```

Mặc định script sẽ:

- gộp `train + val`
- tách lại một phần validation nội bộ để chọn checkpoint
- giữ `X_test.npy` riêng để đánh giá cuối

File huấn luyện chính:

- `src/training/train_best_final.py`

## 6. Đánh giá mô hình

Đánh giá model cuối:

```bash
python scripts/evaluate.py
```

Hoặc chỉ định đường dẫn model:

```bash
python scripts/evaluate.py --model_path models/cnn_lstm_final_best.keras
```

Evaluator sẽ:

- in `Accuracy`
- in `Macro Precision / Recall / F1`
- in `Weighted Precision / Recall / F1`
- in `Per-class Recall`
- in `Classification Report`
- in tóm tắt `Confusion Matrix`

Đồng thời lưu:

- `models/evaluation_final_best_report.json`
- `models/evaluation_final_best_summary.json`
- `models/confusion_matrix_final_best.png`
- `models/confusion_matrix_final_best_normalized.png`

## 7. Chạy Streamlit

### 7.1. Chạy local

Sau khi đã có:

- `models/cnn_lstm_final_best.keras`
- `datasets/processed/label_encoder.pkl`
- `models/mediapipe/hand_landmarker.task`
- `models/mediapipe/face_landmarker.task`

chạy:

```bash
streamlit run app.py
```

App mặc định sẽ dùng:

- model: `models/cnn_lstm_final_best.keras`
- label encoder: `datasets/processed/label_encoder.pkl`

### 7.2. Deploy lên Streamlit Community Cloud

Repo đã được chuẩn bị sẵn các file:

- `app.py`: entrypoint cho Streamlit
- `.streamlit/config.toml`: cấu hình server
- `requirements.txt`: dependency để cài trên môi trường deploy

Khi deploy, chọn:

- **Main file path**: `app.py`

Lưu ý:

- bạn vẫn cần đưa các file model và MediaPipe task cần thiết lên nơi deploy
- app cần ít nhất các file:
  - `models/cnn_lstm_final_best.keras`
  - `datasets/processed/label_encoder.pkl`
  - `models/mediapipe/hand_landmarker.task`
  - `models/mediapipe/face_landmarker.task`

Nếu không muốn public model/dataset lớn, bạn có thể:

- deploy nội bộ / server riêng
- hoặc upload riêng các file model cần thiết lên môi trường chạy

## 8. File đầu ra quan trọng

Sau khi train model cuối, các file chính sẽ nằm trong `models/`:

- `cnn_lstm_final_best.keras`: model tốt nhất
- `cnn_lstm_final_best_history.json`: history huấn luyện
- `cnn_lstm_final_best_metadata.json`: metadata cấu hình train
- `cnn_lstm_final_best_test_report.json`: classification report trên test set

## 9. Luồng làm việc khuyến nghị

Thứ tự chạy khuyến nghị:

1. Chuẩn bị dữ liệu bằng `scripts/prepare_data.py`
2. Dùng `python scripts/train.py loso` để tuning
3. Chốt config tốt nhất
4. Chạy `python scripts/train.py final_best`
5. Chạy `python scripts/evaluate.py`
6. Chạy `streamlit run app.py`

## 10. Ghi chú

- Nên đánh giá bằng `LOSO` trong giai đoạn tuning thay vì chỉ nhìn một split cố định.
- Khi xác nhận một cấu hình tốt, nên chạy nhiều seed để kiểm tra độ ổn định.
- Nếu thêm person hoặc thêm class, hãy chạy lại pipeline `scan -> extract -> preprocess`.
