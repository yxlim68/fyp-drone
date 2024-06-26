from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Results

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("oop.mp4")

def detect_person(result: Results) -> bool:
    # only consider that probs that a person at least 0.8
    min_threshold = 0.8
    name_id = 0  # 0 is person id

    for box in result.boxes:
        conf = box.conf[0].item()
        cls_id = box.cls[0].item()

        if conf >= min_threshold and cls_id == name_id:
            return True

    return False


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to ready frame!")
        continue

    results = model.track(frame, persist=True, verbose=True)

    annotated_frame = results[0].plot()

    person_detected = detect_person(results[0])
    if person_detected:
        print("Person detected")
        cv2.imshow("Detected Frame", annotated_frame)
    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
