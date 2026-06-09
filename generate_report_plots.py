import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure output directory for figures exists
os.makedirs("report_images", exist_ok=True)

# Set global matplotlib style for clean, professional academic look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'grid.color': '#E2E8F0',
    'grid.linewidth': 0.8
})


def generate_split_comparison():
    """
    Hình 1: So sánh Random Split vs Subject-Independent Split
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    np.random.seed(42)

    # Colors representing different subjects (Persons)
    colors = {
        'P1': '#3B82F6',  # Blue
        'P2': '#10B981',  # Emerald Green
        'P3': '#F59E0B',  # Amber/Orange
        'P4': '#EC4899',  # Pink
        'P5': '#8B5CF6'   # Purple
    }

    # Helper function to draw a split column box
    def draw_split_box(ax, x_offset, width, height, title, dot_colors, label):
        # Draw bounding rectangle
        rect = patches.Rectangle((x_offset, 0.1), width, height, linewidth=1.5,
                                 edgecolor='#475569', facecolor='#F8FAFC', zorder=1)
        ax.add_patch(rect)
        ax.text(x_offset + width/2, 0.1 + height + 0.05, title,
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1E293B')
        
        # Scatter random points inside the box
        num_dots = 45
        xs = np.random.uniform(x_offset + 0.05, x_offset + width - 0.05, num_dots)
        ys = np.random.uniform(0.15, 0.1 + height - 0.05, num_dots)
        
        # Assign colors based on list
        for x, y in zip(xs, ys):
            col_key = np.random.choice(dot_colors)
            ax.scatter(x, y, color=colors[col_key], s=55, edgecolors='white', linewidths=0.5, zorder=2)
            
        ax.text(x_offset + width/2, 0.03, label, ha='center', va='top', fontsize=9, color='#64748B')

    # --- PANEL 1: Random Split ---
    ax1.set_title("1. RANDOM SPLIT (Rò rỉ thông tin người thực hiện)", pad=20, fontsize=12, fontweight='bold', color='#B91C1C')
    # Train box contains all persons mixed
    draw_split_box(ax1, 0.05, 0.40, 0.7, "Train (70%)", ['P1', 'P2', 'P3', 'P4', 'P5'], "Chứa P1, P2, P3, P4, P5")
    # Val box contains all persons mixed
    draw_split_box(ax1, 0.50, 0.20, 0.7, "Val (15%)", ['P1', 'P2', 'P3', 'P4', 'P5'], "Chứa P1, P2, P3, P4, P5")
    # Test box contains all persons mixed
    draw_split_box(ax1, 0.75, 0.20, 0.7, "Test (15%)", ['P1', 'P2', 'P3', 'P4', 'P5'], "Chứa P1, P2, P3, P4, P5")
    
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.axis('off')

    # --- PANEL 2: Subject-Independent Split ---
    ax2.set_title("2. SUBJECT-INDEPENDENT SPLIT (Đánh giá khách quan)", pad=20, fontsize=12, fontweight='bold', color='#15803D')
    # Train box contains only P1, P2, P3
    draw_split_box(ax2, 0.05, 0.40, 0.7, "Train (70%)", ['P1', 'P2', 'P3'], "Chỉ chứa P1, P2, P3")
    # Val box contains only P4
    draw_split_box(ax2, 0.50, 0.20, 0.7, "Val (15%)", ['P4'], "Chỉ chứa P4")
    # Test box contains only P5
    draw_split_box(ax2, 0.75, 0.20, 0.7, "Test (15%)", ['P5'], "Chỉ chứa P5")
    
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.axis('off')

    # Add legend for subjects
    legend_elements = [
        patches.Patch(facecolor=colors['P1'], edgecolor='white', label='Person 1'),
        patches.Patch(facecolor=colors['P2'], edgecolor='white', label='Person 2'),
        patches.Patch(facecolor=colors['P3'], edgecolor='white', label='Person 3'),
        patches.Patch(facecolor=colors['P4'], edgecolor='white', label='Person 4'),
        patches.Patch(facecolor=colors['P5'], edgecolor='white', label='Person 5'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("report_images/split_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: report_images/split_comparison.png")


def generate_class_distribution():
    """
    Hình 2: Phân bố samples theo class trong các tập Train/Val/Test
    """
    # Load dataset labels
    processed_path = "datasets/processed"
    y_train_path = os.path.join(processed_path, "y_train.npy")
    y_val_path = os.path.join(processed_path, "y_val.npy")
    y_test_path = os.path.join(processed_path, "y_test.npy")
    le_path = os.path.join(processed_path, "label_encoder.pkl")

    if not (os.path.exists(y_train_path) and os.path.exists(y_val_path) and 
            os.path.exists(y_test_path) and os.path.exists(le_path)):
        print("Error: Processed dataset files not found. Skipping distribution plot.")
        return

    # Load data
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    y_test = np.load(y_test_path)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    classes = list(le.classes_)

    num_classes = len(classes)
    train_counts = [np.sum(y_train == i) for i in range(num_classes)]
    val_counts = [np.sum(y_val == i) for i in range(num_classes)]
    test_counts = [np.sum(y_test == i) for i in range(num_classes)]

    # Plotting grouped horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    y_indices = np.arange(num_classes)
    bar_width = 0.25

    # Grouped bars
    rects_train = ax.barh(y_indices + bar_width, train_counts, bar_width, label='Train Set', color='#3B82F6', edgecolor='none')
    rects_val = ax.barh(y_indices, val_counts, bar_width, label='Val Set', color='#10B981', edgecolor='none')
    rects_test = ax.barh(y_indices - bar_width, test_counts, bar_width, label='Test Set', color='#F59E0B', edgecolor='none')

    # Styling labels and layout
    ax.set_xlabel('Số lượng mẫu (Samples)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Danh mục ngôn ngữ ký hiệu (Classes)', fontweight='bold', labelpad=10)
    ax.set_title('PHÂN BỐ SỐ LƯỢNG MẪU THEO TỪNG LỚP TRONG CÁC TẬP DATA SPLIT', fontweight='bold', pad=15)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(classes, fontweight='bold', color='#1E293B')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Helper to add value labels on bars
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate(f'{width}',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9, color='#334155')

    autolabel(rects_train)
    autolabel(rects_val)
    autolabel(rects_test)

    # Show summary of dataset shape on bottom
    total_train = len(y_train)
    total_val = len(y_val)
    total_test = len(y_test)
    total_all = total_train + total_val + total_test
    summary_text = (f"Tổng mẫu: {total_all}  |  "
                    f"Train: {total_train} ({total_train/total_all*100:.1f}%)  |  "
                    f"Val: {total_val} ({total_val/total_all*100:.1f}%)  |  "
                    f"Test: {total_test} ({total_test/total_all*100:.1f}%)")
    plt.figtext(0.5, 0.02, summary_text, ha="center", fontsize=10, fontweight="bold",
                bbox={"facecolor": "#F1F5F9", "alpha": 0.8, "pad": 6, "edgecolor": "#CBD5E1"})

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig("report_images/class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: report_images/class_distribution.png")


def generate_augmentation_comparison():
    """
    Hình 3: Biểu đồ minh họa Augmentation (Original + 4 biến thể)
    """
    frames = np.arange(30)
    # Original signal representing smooth hand gesture movement (e.g. wrist x-coordinate)
    original = np.sin(frames * 0.2) * 1.5 + 0.1 * frames

    # 1. Jittering (Noise)
    np.random.seed(24)
    noise = np.random.normal(0, 0.15, size=30)
    jittered = original + noise

    # 2. Time Warp (Speed adjustment)
    # Stretched (Slower) - take indices spread out
    stretched_indices = np.linspace(0, 20, 30)
    stretched = np.sin(stretched_indices * 0.2) * 1.5 + 0.1 * stretched_indices

    # 3. Scaling (Scaling size/amplitude)
    scaled = original * 0.7

    # 4. Shift (Spatial translation)
    shifted = original + 0.5

    # Plot 1x5 subplots side-by-side to compare clearly
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)

    titles = [
        "Original (Gốc)",
        "Jittering (+Nhiễu)",
        "Time Warp (Tốc độ)",
        "Scaling (Co giãn)",
        "Shift (Dịch vị trí)"
    ]

    datasets = [original, jittered, stretched, scaled, shifted]
    colors = ['#1E293B', '#EF4444', '#3B82F6', '#10B981', '#8B5CF6']

    for i, (ax, title, data, color) in enumerate(zip(axes, titles, datasets, colors)):
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # If not the original subplot, draw the original line as a dashed gray line for reference
        if i > 0:
            ax.plot(frames, original, color='#94A3B8', linestyle='--', linewidth=1.5, label='Original')
            ax.plot(frames, data, color=color, linewidth=2.5, label='Augmented')
        else:
            ax.plot(frames, original, color=color, linewidth=2.5, label='Original')
            
        ax.set_title(title, fontsize=11, fontweight='bold', color='#1E293B')
        ax.set_xlabel("Chỉ số Frame", fontsize=9)
        if i == 0:
            ax.set_ylabel("Giá trị Tọa độ (Normalized)", fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.legend(loc='upper left', fontsize=8)
            
        ax.set_ylim(-1, 5)

    plt.suptitle("SO SÁNH CÁC PHƯƠNG PHÁP TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION) TRÊN CHUỖI LANDMARK", 
                 y=1.05, fontsize=13, fontweight='bold', color='#1E293B')
    plt.tight_layout()
    plt.savefig("report_images/augmentation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: report_images/augmentation_comparison.png")


if __name__ == "__main__":
    print("Starting generation of report illustration images...")
    generate_split_comparison()
    generate_class_distribution()
    generate_augmentation_comparison()
    print("All illustration images generated successfully inside folder 'report_images/'.")
