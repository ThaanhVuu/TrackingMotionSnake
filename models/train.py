import multiprocessing
from ultralytics import YOLO


def main():
    # 🧩 Load model
    model = YOLO("yolo11n.pt")

    # 🚀 Train
    results = model.train(
        data="../dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        cache=True,
        workers=4,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Bắt buộc trên Windows
    main()
