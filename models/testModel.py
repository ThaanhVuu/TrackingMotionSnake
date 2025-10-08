from ultralytics import YOLO

# Load model đã train
model = YOLO("../runs/detect/train/weights/best.pt")

# Dự đoán ảnh
results = model.predict(source="../input/image.jpg", conf=0.1, save=True, show=True)
# source: có thể là 1 file ảnh, thư mục, video, webcam, hoặc URL
