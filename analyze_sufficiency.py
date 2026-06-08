import os
from collections import Counter

root = 'e:/h1eesu/AI/sign_language/datasets/dataset'

# Define splits
TRAIN_PERSONS = ['person_01', 'person_02', 'person_03']
VAL_PERSONS   = ['person_04']
TEST_PERSONS  = ['person_05', 'person_06']
SKIP_PERSONS  = []

# Merge pain/hurt
LABEL_MAP = {'hurt': 'pain'}

# Collect data
splits = {'train': Counter(), 'val': Counter(), 'test': Counter(), 'skipped': Counter()}
total_per_label = Counter()

for person in sorted(os.listdir(root)):
    ppath = os.path.join(root, person)
    if not os.path.isdir(ppath):
        continue

    if person in SKIP_PERSONS:
        split_name = 'skipped'
    elif person in TRAIN_PERSONS:
        split_name = 'train'
    elif person in VAL_PERSONS:
        split_name = 'val'
    elif person in TEST_PERSONS:
        split_name = 'test'
    else:
        continue

    for label in os.listdir(ppath):
        lpath = os.path.join(ppath, label)
        if not os.path.isdir(lpath):
            continue
        label_clean = LABEL_MAP.get(label, label)
        n_vids = len([f for f in os.listdir(lpath) if os.path.isfile(os.path.join(lpath, f))])
        splits[split_name][label_clean] += n_vids
        if split_name != 'skipped':
            total_per_label[label_clean] += n_vids

# All labels
all_labels = sorted(total_per_label.keys())

print("=" * 80)
print("PHÂN TÍCH ĐỘ ĐẦY ĐỦ CỦA DATASET")
print("=" * 80)

# Summary
total_train = sum(splits['train'].values())
total_val = sum(splits['val'].values())
total_test = sum(splits['test'].values())
total_all = total_train + total_val + total_test

print(f"\nTổng video (sau loại P07-09): {total_all}")
print(f"  Train (P01-03): {total_train} ({100*total_train/total_all:.1f}%)")
print(f"  Val   (P04):    {total_val} ({100*total_val/total_all:.1f}%)")
print(f"  Test  (P05-06): {total_test} ({100*total_test/total_all:.1f}%)")
print(f"  Số class:       {len(all_labels)}")

# Per-class breakdown
print(f"\n{'Label':15s} {'Train':>7s} {'Val':>5s} {'Test':>6s} {'Total':>7s} {'Aug(×6)':>8s}  Status")
print("-" * 70)

issues = []
for label in all_labels:
    tr = splits['train'][label]
    va = splits['val'][label]
    te = splits['test'][label]
    tot = tr + va + te
    aug = tr * 6

    # Check issues
    status = "✅"
    if tr == 0:
        status = "❌ No train data!"
        issues.append((label, "Không có dữ liệu train"))
    elif tr < 3:
        status = "🔴 Very few train"
        issues.append((label, f"Chỉ {tr} video train"))
    elif va == 0:
        status = "⚠️ No val data"
        issues.append((label, "Không có dữ liệu validation"))
    elif te == 0:
        status = "⚠️ No test data"
        issues.append((label, "Không có dữ liệu test"))
    elif tr < 5:
        status = "🟡 Low train"

    print(f"  {label:15s} {tr:5d}   {va:3d}   {te:4d}   {tot:5d}   {aug:6d}    {status}")

# Statistics
train_counts = [splits['train'][l] for l in all_labels]
print(f"\n--- Thống kê Train set ---")
print(f"  Min videos/class:  {min(train_counts)}")
print(f"  Max videos/class:  {max(train_counts)}")
print(f"  Avg videos/class:  {sum(train_counts)/len(train_counts):.1f}")
print(f"  Sau augment (×6):  min={min(train_counts)*6}, avg={sum(train_counts)/len(train_counts)*6:.0f}")

# Labels missing in val/test
val_missing = [l for l in all_labels if splits['val'][l] == 0]
test_missing = [l for l in all_labels if splits['test'][l] == 0]
train_missing = [l for l in all_labels if splits['train'][l] == 0]

if val_missing:
    print(f"\n⚠️  Labels THIẾU trong Val:  {val_missing}")
if test_missing:
    print(f"⚠️  Labels THIẾU trong Test: {test_missing}")
if train_missing:
    print(f"❌ Labels THIẾU trong Train: {train_missing}")

# Class imbalance ratio
if train_counts:
    imbalance = max(train_counts) / max(1, min(train_counts))
    print(f"\n--- Imbalance ratio (Train) ---")
    print(f"  Max/Min = {max(train_counts)}/{min(train_counts)} = {imbalance:.1f}x")
    if imbalance > 5:
        print(f"  ⚠️  Imbalance > 5x — nên dùng class weights hoặc oversampling")
    elif imbalance > 3:
        print(f"  🟡 Imbalance vừa phải — có thể chấp nhận được")
    else:
        print(f"  ✅ Imbalance OK")

# Benchmark comparison
print(f"\n{'='*80}")
print(f"SO SÁNH VỚI CÁC DATASET BENCHMARK")
print(f"{'='*80}")
print(f"""
  Dataset gốc    :  {total_all} videos, {len(all_labels)} classes
  Sau augment ×6 :  ~{total_train*6} train samples

  Tham khảo các bài toán tương tự (landmark-based gesture recognition):
  ┌────────────────────────────────┬──────────┬─────────┬──────────────┐
  │ Dataset                        │ Samples  │ Classes │ Accuracy     │
  ├────────────────────────────────┼──────────┼─────────┼──────────────┤
  │ SHREC'17 (14 gestures)         │    2,800 │      14 │ 93-95%       │
  │ DHG-14/28                      │    2,800 │   14/28 │ 85-92%       │
  │ WLASL (ASL, 100 words)         │   ~2,000 │     100 │ 60-65%       │
  │ AUTSL (Turkish SL)             │   36,302 │     226 │ 90%+         │
  │ 👉 Bạn (sau augment)           │   ~{total_train*6:,} │      {len(all_labels)} │ ???          │
  └────────────────────────────────┴──────────┴─────────┴──────────────┘

  → Với ~{total_train*6:,} samples / {len(all_labels)} classes ≈ {total_train*6//len(all_labels)} samples/class
    Đây là mức CHẤP NHẬN ĐƯỢC cho landmark-based classification.
""")

print(f"{'='*80}")
print(f"KẾT LUẬN")
print(f"{'='*80}")
print(f"""
  ✅ CÓ THỂ TRAIN ĐƯỢC — nhưng cần lưu ý:

  1. Dữ liệu KHÔNG THỪA — mỗi sample đều quý giá
     → Augmentation là BẮT BUỘC, không phải optional

  2. Một số class ít dữ liệu ({min(train_counts)} video train)
     → Nên dùng class_weight trong loss function

  3. Landmark-based (63 features) nhẹ hơn image-based
     → Cần ÍT dữ liệu hơn so với CNN trên raw frame
     → LSTM/CNN1D+LSTM phù hợp cho kích thước dataset này

  4. Subject-independent split sẽ cho accuracy THẤP HƠN random split
     → Accuracy 75-88% là KẾT QUẢ TỐT cho setup này
     → Đừng so sánh với paper dùng random split (95%+)
""")
