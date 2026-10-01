import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

TOTAL_SEATS = 10  # change this to your actual test seat count

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame.")
        break

    results = model(frame, classes=[0], verbose=False)

    person_count = 0
    annotated_frame = frame
    for result in results:
        person_count = len(result.boxes)
        annotated_frame = result.plot()

    empty_seats = max(0, TOTAL_SEATS - person_count)

    cv2.putText(annotated_frame, f"Students detected: {person_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Empty seats: {empty_seats}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Seat Detection Test", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()