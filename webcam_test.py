import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        # MediaPipe needs RGB, OpenCV gives BGR — convert it
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        face_count = 0
        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Show live face count on screen
        cv2.putText(frame, f"Faces detected: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()