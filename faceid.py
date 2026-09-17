import cv2
import os
import sys
import numpy as np
import json
import threading
from datetime import datetime
from deepface import DeepFace

# Directory configuration
recognized_faces_dir = "recognized_faces"
os.makedirs(recognized_faces_dir, exist_ok=True)
info_file = "recognized_faces_info.json"

window_width = 800
window_height = 600

# Load info file
face_info = {}
try:
    with open(info_file, 'r') as f:
        content = f.read()
        if content.strip():
            face_info = json.loads(content)
except (json.JSONDecodeError, FileNotFoundError):
    pass

# Load saved faces into memory
face_cache = {}
def load_face_cache():
    global face_cache
    face_cache = {}
    for filename in os.listdir(recognized_faces_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(recognized_faces_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                face_cache[filename] = img

load_face_cache()

# Cross-platform sound alert
def play_alert():
    if sys.platform == "win32":
        import winsound
        winsound.Beep(1000, 300)
    elif sys.platform == "darwin":
        os.system('afplay /System/Library/Sounds/Ping.aiff &')

def draw_multiline_text(img, text, origin, font, scale, color, thickness, line_spacing=22):
    x, y = origin
    for i, line in enumerate(text.split('\n')):
        cv2.putText(img, line, (x, y + (i * line_spacing)), font, scale, color, thickness)

cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def initialize_camera():
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else (cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    for index in (0, 1):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    return cv2.VideoCapture(0)

video_capture = initialize_camera()

def is_face_recognized(face_img):
    for filename, saved_face in face_cache.items():
        saved_resized = cv2.resize(saved_face, (face_img.shape[1], face_img.shape[0]))
        difference = cv2.absdiff(saved_resized, face_img)
        if np.mean(difference) < 50:
            return filename
    return None

# Threading state variables
analysis_results = {}
analysis_lock = threading.Lock()
is_analyzing = False

def run_deepface_async(face_crop, face_key):
    """Background worker function for heavy model inference."""
    global is_analyzing
    try:
        face_img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        result = DeepFace.analyze(face_img_rgb, actions=['age', 'gender'], enforce_detection=False)
        age = result[0]['age']
        gender = result[0].get('dominant_gender', result[0]['gender'])

        with analysis_lock:
            analysis_results[face_key] = (age, gender)
    except Exception:
        pass
    finally:
        is_analyzing = False

# Main video loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for idx, (x, y, w, h) in enumerate(faces):
        face_img = gray[y:y+h, x:x+w]
        face_key = f"face_{idx}"

        recognized_filename = is_face_recognized(face_img)
        if recognized_filename:
            status_text = "Recognized"
            color = (0, 255, 0)
            face_id = os.path.splitext(recognized_filename)[0]
            person_info = face_info.get(face_id, {})
            name = person_info.get("name", "Unknown")
            last_seen = person_info.get("last_seen", "Never")
            additional_info = f"Name: {name}\nLast Seen: {last_seen}"
        else:
            face_id = f"face_{len(face_cache) + 1}"
            face_filename = f"{face_id}.jpg"
            file_path = os.path.join(recognized_faces_dir, face_filename)
            cv2.imwrite(file_path, face_img)

            # Update cache immediately
            face_cache[face_filename] = face_img

            person_info = {
                "name": f"Person {len(face_info) + 1}",
                "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            face_info[face_id] = person_info
            with open(info_file, 'w') as f:
                json.dump(face_info, f, indent=4)

            status_text = "Not Recognized"
            color = (0, 0, 255)
            additional_info = "New face saved."
            play_alert()

        # Trigger background analysis if a thread isn't currently active
        if not is_analyzing:
            is_analyzing = True
            threading.Thread(
                target=run_deepface_async,
                args=(face_img.copy(), face_key),
                daemon=True
            ).start()

        # Retrieve cached DeepFace stats safely across threads
        with analysis_lock:
            age, gender = analysis_results.get(face_key, ("Analyzing...", "Analyzing..."))

        additional_info += f"\nAge: {age}, Gender: {gender}"

        # Render overlays
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        draw_multiline_text(frame, additional_info, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    resized_frame = cv2.resize(frame, (window_width, window_height))
    cv2.imshow('Face Recognition', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
