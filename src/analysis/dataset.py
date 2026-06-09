import os

# pyrefly: ignore [missing-import]
from src.config import DATASET_ROOT

label_counts = {}
person_label = {}

for person in sorted(os.listdir(DATASET_ROOT)):
    ppath = os.path.join(DATASET_ROOT, person)
    if not os.path.isdir(ppath):
        continue
    for label in sorted(os.listdir(ppath)):
        lpath = os.path.join(ppath, label)
        if not os.path.isdir(lpath):
            continue
        vids = [f for f in os.listdir(lpath) if os.path.isfile(os.path.join(lpath, f))]
        label_counts[label] = label_counts.get(label, 0) + len(vids)
        if label not in person_label:
            person_label[label] = {}
        person_label[label][person] = len(vids)

print("=== Videos per label (total) ===")
for l in sorted(label_counts.keys()):
    print(f"  {l:15s}: {label_counts[l]:3d} videos")

print(f"\nTotal labels: {len(label_counts)}")
print(f"Total videos: {sum(label_counts.values())}")

# Show labels per person
print("\n=== Labels per person ===")
persons = sorted(set(p for ld in person_label.values() for p in ld))
header = f"{'label':15s}" + "".join(f"{p:>12s}" for p in persons)
print(header)
for l in sorted(person_label.keys()):
    row = f"{l:15s}"
    for p in persons:
        cnt = person_label[l].get(p, 0)
        row += f"{cnt:12d}"
    print(row)

# Check which persons have incomplete labels
print("\n=== Persons with missing labels ===")
all_labels = set(label_counts.keys())
for p in persons:
    p_labels = set(l for l, pd in person_label.items() if p in pd)
    missing = all_labels - p_labels
    if missing:
        print(f"  {p}: missing {len(missing)} labels -> {sorted(missing)}")
    else:
        print(f"  {p}: all {len(all_labels)} labels present")
