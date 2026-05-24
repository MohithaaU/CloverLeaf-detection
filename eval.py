from ultralytics import YOLO

# Load your model
model = YOLO("best.pt")

# Run evaluation on test dataset
metrics = model.val(
    data="data.yaml",
    split="test"  # uses test folder specifically
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)