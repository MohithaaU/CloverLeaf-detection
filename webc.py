import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("bestv2.pt")

# Open webcam
cap = cv2.VideoCapture(0)

print("Webcam started! Press Q to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame, conf=0.35, iou=0.3)
    
    # Draw boxes on frame
    annotated_frame = results[0].plot()
    
    # Show the frame
    cv2.imshow("Clover Detector 🍀", annotated_frame)
    
    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()