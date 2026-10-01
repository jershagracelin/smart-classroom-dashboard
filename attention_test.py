import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Actual resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_face_mesh.FaceMesh(max_num_faces=20, refine_landmarks=True,
                            min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        attentive_count = 0
        not_attentive_count = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_tip = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                eye_mid_x = (left_eye.x + right_eye.x) / 2
                offset = nose_tip.x - eye_mid_x

                if abs(offset) < 0.03:
                    status = "Attentive"
                    color = (0, 255, 0)
                    attentive_count += 1
                else:
                    status = "Not Attentive"
                    color = (0, 0, 255)
                    not_attentive_count += 1

                x_pixel = int(nose_tip.x * w)
                y_pixel = int(nose_tip.y * h)
                cv2.putText(frame, status, (x_pixel - 40, y_pixel - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total = attentive_count + not_attentive_count
        attention_pct = int((attentive_count / total) * 100) if total > 0 else 0

        cv2.putText(frame, f"Attentive: {attentive_count}  Not Attentive: {not_attentive_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Attention: {attention_pct}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Attention Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()