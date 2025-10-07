import multiprocessing
import torch
from ultralytics import YOLO


def main():
    # 🔍 Kiểm tra GPU khả dụng
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ Có {device_count} GPU khả dụng.")

        # 🧠 Tự động chọn GPU rảnh (ít memory usage nhất)
        min_used = None
        selected_gpu = 0
        for i in range(device_count):
            mem_info = torch.cuda.mem_get_info(i)
            free_mem = mem_info[0] / mem_info[1]  # phần trăm bộ nhớ trống
            print(f"GPU {i}: {free_mem * 100:.2f}% bộ nhớ trống.")
            if (min_used is None) or (free_mem > min_used):
                min_used = free_mem
                selected_gpu = i
        device = selected_gpu
        print(f"🎯 Sử dụng GPU {device} để train.")
    else:
        device = "cpu"
        print("⚠️ Không phát hiện GPU, sẽ train bằng CPU (rất chậm).")

    # 🧩 Load model
    model = YOLO("yolo11n.pt")

    # 🚀 Train
    results = model.train(
        data="../dataset/data.yaml",
        epochs=100,
        imgsz=1280,
        device=device,  # GPU tự chọn
        workers=0,  # giảm lỗi multiprocessing trên Windows
        name="train_auto",  # tên run
        deterministic=True  # kết quả ổn định
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Bắt buộc trên Windows
    main()
