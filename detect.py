from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("bestv4.pt")
# Run detection on your image
results = model("t2.jpg",conf=0.5,iou=0.3)

# Show the result
results[0].show()

metrics = model.val(
    data="data.yaml",
    split="test"  # uses test folder specifically
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)