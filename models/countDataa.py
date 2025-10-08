
import os

# 📁 Đường dẫn gốc dataset của bạn
DATASET_DIR = "../dataset"

def count_files_in_dir(path, exts):
    """Đếm số file có đuôi trong danh sách exts"""
    return sum(
        1 for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(tuple(exts))
    )

def summarize_split(split_name):
    """Thống kê số ảnh và số label trong train / valid / test"""
    img_dir = os.path.join(DATASET_DIR, split_name, "images")
    lbl_dir = os.path.join(DATASET_DIR, split_name, "labels")

    num_images = count_files_in_dir(img_dir, [".jpg", ".jpeg", ".png"])
    num_labels = count_files_in_dir(lbl_dir, [".txt"])

    print(f"📂 {split_name.upper()}:")
    print(f"   🖼️  Số ảnh   : {num_images}")
    print(f"   🏷️  Số nhãn  : {num_labels}")
    print(f"   ⚖️  Chênh lệch: {abs(num_images - num_labels)} file\n")

def main():
    print("📊 Thống kê dataset YOLO\n---------------------------")
    for split in ["train", "valid", "test"]:
        summarize_split(split)
    print("✅ Kiểm tra hoàn tất!")

if __name__ == "__main__":
    main()
