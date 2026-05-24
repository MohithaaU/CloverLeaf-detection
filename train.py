from ultralytics import YOLO

# Load the smallest YOLOv8 model (fastest on CPU)
model = YOLO("yolov8n.pt")

# Train it on our clover dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    name="clover_detector"
)

print("Training done!")