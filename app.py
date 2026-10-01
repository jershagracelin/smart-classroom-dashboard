from flask import Flask, render_template, jsonify, request
import cv2
import mediapipe as mp
import pyaudio
import webrtcvad
import threading
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

DB_FILE = "session_log.db"

dashboard_data = {
    "attention_pct": 0,
    "students_present": 0,
    "empty_seats": 0,
    "noise_level": "Low",
    "speaking_time_min": 0,
    "engagement_score": 0,
    "alert_level": "normal",
    "alert_message": ""
}

TOTAL_SEATS = 10
mp_face_mesh = mp.solutions.face_mesh
yolo_model = YOLO("yolov8n.pt")

speaking_seconds = 0
total_seconds = 0
audio_lock = threading.Lock()


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            attention_pct INTEGER,
            students_present INTEGER,
            empty_seats INTEGER,
            noise_level TEXT,
            speaking_time_min REAL,
            engagement_score REAL,
            alert_level TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_session_data(data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO session_data
        (timestamp, attention_pct, students_present, empty_seats, noise_level, speaking_time_min, engagement_score, alert_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["attention_pct"],
        data["students_present"],
        data["empty_seats"],
        data["noise_level"],
        data["speaking_time_min"],
        data["engagement_score"],
        data["alert_level"]
    ))
    conn.commit()
    conn.close()


def calculate_engagement_score(attention_pct, empty_seats, total_seats, noise_level):
    attention_component = attention_pct * 0.5
    occupancy_pct = ((total_seats - empty_seats) / total_seats) * 100 if total_seats > 0 else 0
    occupancy_component = occupancy_pct * 0.3
    noise_score = {"Low": 100, "Medium": 60, "High": 20}.get(noise_level, 50)
    noise_component = noise_score * 0.2
    return round(attention_component + occupancy_component + noise_component, 1)


def check_alerts(engagement_score, noise_level, attention_pct):
    if noise_level == "High" and attention_pct < 50:
        return "critical", "High noise AND low attention detected!"
    elif engagement_score < 50:
        return "warning", "Engagement score is low."
    elif noise_level == "High":
        return "warning", "Classroom noise is high."
    else:
        return "normal", "All good."


def video_processing_loop():
    """Handles webcam-based attention + YOLO seat/person detection, and periodic SQLite logging."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_log_time = [time.time()]

    with mp_face_mesh.FaceMesh(max_num_faces=20, refine_landmarks=True,
                                min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

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
                        attentive_count += 1
                    else:
                        not_attentive_count += 1

            total_faces = attentive_count + not_attentive_count
            attention_pct = int((attentive_count / total_faces) * 100) if total_faces > 0 else 0

            # ---- YOLO person/seat detection ----
            yolo_results = yolo_model(frame, classes=[0], verbose=False)
            person_count = 0
            for result in yolo_results:
                person_count = len(result.boxes)
            # -------------------------------------

            empty_seats = max(0, TOTAL_SEATS - person_count)

            with audio_lock:
                speaking_min = round(speaking_seconds / 60, 1)

            dashboard_data["attention_pct"] = attention_pct
            dashboard_data["students_present"] = person_count
            dashboard_data["empty_seats"] = empty_seats
            dashboard_data["speaking_time_min"] = speaking_min

            engagement_score = calculate_engagement_score(
                attention_pct, empty_seats, TOTAL_SEATS, dashboard_data["noise_level"]
            )
            alert_level, alert_message = check_alerts(
                engagement_score, dashboard_data["noise_level"], attention_pct
            )
            dashboard_data["engagement_score"] = engagement_score
            dashboard_data["alert_level"] = alert_level
            dashboard_data["alert_message"] = alert_message

            if time.time() - last_log_time[0] >= 10:
                log_session_data(dashboard_data)
                last_log_time[0] = time.time()

            time.sleep(0.5)

    cap.release()


def audio_processing_loop():
    """Handles microphone-based speaking time detection (VAD)."""
    global speaking_seconds, total_seconds

    vad = webrtcvad.Vad(2)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    FRAME_DURATION = 30
    FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                             input=True, frames_per_buffer=FRAME_SIZE)
    except Exception as e:
        print(f"ERROR: Could not open microphone: {e}")
        return

    print("Speaking time detection started.")

    while True:
        try:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)

            with audio_lock:
                total_seconds += FRAME_DURATION / 1000
                if is_speech:
                    speaking_seconds += FRAME_DURATION / 1000
        except Exception as e:
            print(f"ERROR in audio loop: {e}")
            time.sleep(1)


@app.route("/")
def index():
    return render_template("dashboard.html", data=dashboard_data)


@app.route("/api/data")
def get_data():
    return jsonify(dashboard_data)


@app.route("/api/history")
def get_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM session_data ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    columns = ["id", "timestamp", "attention_pct", "students_present", "empty_seats",
               "noise_level", "speaking_time_min", "engagement_score", "alert_level"]
    return jsonify([dict(zip(columns, row)) for row in rows])


@app.route("/api/esp32", methods=["POST"])
def receive_esp32_data():
    data = request.get_json()
    sound_value = data.get("sound_value", 0)
    print(f"Received from ESP32: {sound_value}")

    if sound_value > 2500:
        dashboard_data["noise_level"] = "High"
    elif sound_value > 1500:
        dashboard_data["noise_level"] = "Medium"
    else:
        dashboard_data["noise_level"] = "Low"

    return jsonify({"status": "received"})


if __name__ == "__main__":
    init_db()

    video_thread = threading.Thread(target=video_processing_loop, daemon=True)
    video_thread.start()

    audio_thread = threading.Thread(target=audio_processing_loop, daemon=True)
    audio_thread.start()

    app.run(host="0.0.0.0", debug=False)     