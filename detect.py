from ultralytics import YOLO
import cv2

model = YOLO("bestv4.pt")

results = model("t2.jpg",conf=0.5,iou=0.3)

results[0].show()

metrics = model.val(
    data="data.yaml",
    split="test"  
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)
