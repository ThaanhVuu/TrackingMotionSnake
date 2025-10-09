from ultralytics import YOLO
from tkinter import Tk, filedialog
from PIL import Image
from moviepy.editor import VideoFileClip
import os

import ultralytics.trackers.byte_tracker as bt

# ✅ Vá triệt để lỗi thiếu fuse_score trong ByteTrack
orig_update = bt.BYTETracker.update

def patched_update(self, *args, **kwargs):
    # Đảm bảo self.args tồn tại
    if not hasattr(self, "args") or self.args is None:
        from types import SimpleNamespace
        self.args = SimpleNamespace()
    # Đảm bảo có thuộc tính fuse_score
    if not hasattr(self.args, "fuse_score"):
        self.args.fuse_score = True
    return orig_update(self, *args, **kwargs)

bt.BYTETracker.update = patched_update
print("✅ ByteTrack fuse_score hotfix (instance-level) applied!")

# Ẩn cửa sổ chính Tkinter
Tk().withdraw()

# Mở cửa sổ chọn file (ảnh hoặc video)
file_path = filedialog.askopenfilename(
    title="Chọn ảnh hoặc video để dự đoán",
    filetypes=[
        ("Media files", "*.jpg *.jpeg *.png *.webp *.bmp *.mp4 *.avi *.mov *.mkv"),
        ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
    ]
)

if not file_path:
    print("❌ Không chọn file nào!")
    exit()

print("📸 File đã chọn:", file_path)

# Nếu là file ảnh .webp → chuyển sang JPEG
if file_path.lower().endswith(".webp"):
    converted_path = os.path.splitext(file_path)[0] + "_temp.jpg"
    Image.open(file_path).convert("RGB").save(converted_path, "JPEG")
    file_path = converted_path

# Load model
model = YOLO("../models/tuan.pt")

# Thực hiện tracking
results = model.track(
    source=file_path,
    conf=0.3,
    save=True,
    show=True,
    tracker="bytetrack.yaml"
)

# Thư mục lưu kết quả
save_dir = model.predictor.save_dir
print("✅ Dự đoán xong! Kết quả lưu trong thư mục:", save_dir)

# Nếu là video → convert AVI sang MP4
if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
    output_avi = os.path.join(save_dir, f"{base_name}.avi")
    output_mp4 = os.path.join(save_dir, f"{base_name}.mp4")

    if os.path.exists(output_avi):
        print("🎬 Đang chuyển AVI → MP4 ...")
        try:
            clip = VideoFileClip(output_avi)
            clip.write_videofile(output_mp4, codec="libx264", audio=False)
            clip.close()
            os.remove(output_avi)  # Xóa file AVI cũ cho gọn
            print("✅ Video MP4 đã lưu tại:", output_mp4)
        except Exception as e:
            print("⚠️ Lỗi khi chuyển sang MP4:", e)
    else:
        print("⚠️ Không tìm thấy file AVI để chuyển.")
