import multiprocessing
from ultralytics import YOLO

def main():
    model = YOLO('../models/last.pt')

    result = model.train(
        resume=True,
        data='../dataset/data.yaml',
        batch=4,
        device=0,
        workers=2,
        exist_ok=True,
        amp=True,
        cache=True
                        )

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Bắt buộc trên Windows
    main()