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

Tạo môi trường và cài thư viện theo cách bạn đang dùng cho project.
```powershell
pip intall virtualenv
```

Sau đó tạo môi trường ảo
```powershell
python -m venv venv
```
 Nếu đã có `venv` thì chỉ cần kích hoạt môi trường trước khi chạy.

Ví dụ trên Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Khi kích hoạt môi trường ảo thành côn, cài đặt các thư viện cần thiết
```powershell
pip install -r requirements.txt
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
python scripts/train.py final
```

Ví dụ:

```bash
python scripts/train.py final --epochs 150 --seed 42
python scripts/train.py final --epochs 150 --seed 123
```

Mặc định script sẽ:

- gộp `train + val`
- tách lại một phần validation nội bộ để chọn checkpoint
- giữ `X_test.npy` riêng để đánh giá cuối

File huấn luyện chính:

- `src/training/train_final.py`

## 6. Đánh giá mô hình

Đánh giá model cuối:

```bash
python scripts/evaluate.py
```

Hoặc chỉ định đường dẫn model:

```bash
python scripts/evaluate.py --model_path models/cnn_lstm_final.keras
```

Evaluator sẽ:

- in `Accuracy`
- in `Macro Precision / Recall / F1`
- in `Weighted Precision / Recall / F1`
- in `Per-class Recall`
- in `Classification Report`
- in tóm tắt `Confusion Matrix`

Đồng thời lưu:

- `models/evaluation_final_report.json`
- `models/evaluation_final_summary.json`
- `models/confusion_matrix_final.png`
- `models/confusion_matrix_final_normalized.png`

## 7. Chạy Streamlit

### 7.1. Chạy local (Không cần TURN server ngoài)

chạy:

```bash
streamlit run app_local.py
# Hoặc nếu sử dụng uv:
uv run streamlit run app_local.py
```

App local mặc định sẽ dùng:

- script gốc: `src/apps/streamlit_app_local.py`
- model: `models/cnn_lstm_final.keras`
- label encoder: `datasets/processed/label_encoder.pkl`

### 7.2. Deploy lên Streamlit Community Cloud

File `app.py` là entrypoint dành riêng khi deploy lên Streamlit Cloud. Bản này tích hợp các cơ chế nâng cao:

- **Tự động sửa lỗi OpenCV nhị phân (`libgthread-2.0.so.0`):** Tự động tải bản `opencv-python-headless` tương thích vào thư mục `libs/` cục bộ tại runtime để vượt qua lỗi thiếu thư viện của server Streamlit.
- **Tích hợp Metered.ca TURN API:** Tự động gọi API của Metered.ca để lấy thông tin TURN/STUN credentials tạm thời giúp người dùng kết nối camera từ xa qua mạng Internet (traversal qua NAT/Firewall).

Khi deploy lên Streamlit Cloud, cấu hình:

- **Main file path**: `app.py`

Lưu ý các file cần thiết trên Git:

- `models/cnn_lstm_final.keras`
- `datasets/processed/label_encoder.pkl`
- `models/mediapipe/hand_landmarker.task`
- `models/mediapipe/face_landmarker.task`

## 8. File đầu ra quan trọng

Sau khi train model cuối, các file chính sẽ nằm trong `models/`:

- `cnn_lstm_final.keras`: model tốt nhất
- `cnn_lstm_final_history.json`: history huấn luyện
- `cnn_lstm_final_metadata.json`: metadata cấu hình train
- `cnn_lstm_final_test_report.json`: classification report trên test set

## 9. Luồng làm việc khuyến nghị

Thứ tự chạy khuyến nghị:

1. Chuẩn bị dữ liệu bằng `scripts/prepare_data.py` (hoặc dùng `uv run python...`)
2. Dùng `python scripts/train.py loso` để tuning
3. Chốt config tốt nhất
4. Chạy `python scripts/train.py final` (thêm `--train-on-all` nếu muốn train toàn bộ dữ liệu của cả 5 người để mang đi deploy)
5. Chạy `python scripts/evaluate.py`
6. Chạy local để test: `streamlit run app_local.py`
7. Push code lên GitHub để Streamlit Cloud tự cập nhật ứng dụng `app.py`

## 10. Ghi chú

- Nên đánh giá bằng `LOSO` trong giai đoạn tuning thay vì chỉ nhìn một split cố định.
- Khi xác nhận một cấu hình tốt, nên chạy nhiều seed để kiểm tra độ ổn định.
- Nếu thêm person hoặc thêm class, hãy chạy lại pipeline `scan -> extract -> preprocess`.
- Để tránh lỗi đè file đang bị chiếm dụng trên Windows (OSError [Errno 22]), hãy **tắt Streamlit trước khi chạy lại lệnh train**.