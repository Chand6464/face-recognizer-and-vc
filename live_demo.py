import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import threading
import cv2
from deepface import DeepFace

# Limit TensorFlow CPU use so the PC stays usable
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

ANALYZE_WIDTH = 320      # analyze a small copy of the frame
PAUSE_SECONDS = 1.0      # rest between analyses (raise to 2.0 for slower PCs)

latest_frame = None
results = []
scale = 1.0
running = True
lock = threading.Lock()


def worker():
    global results, scale
    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (ANALYZE_WIDTH, int(h * ANALYZE_WIDTH / w)))
        try:
            res = DeepFace.analyze(
                small,
                actions=["age", "gender", "emotion"],
                detector_backend="opencv",
                enforce_detection=False,
            )
        except Exception as e:
            print("Analysis error:", e)
            res = []

        with lock:
            results = res
            scale = w / ANALYZE_WIDTH
        time.sleep(PAUSE_SECONDS)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

threading.Thread(target=worker, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame
        current, s = results, scale

    for face in current:
        if face.get("face_confidence", 1) == 0:
            continue  # no real face found
        r = face["region"]
        x, y = int(r["x"] * s), int(r["y"] * s)
        w, h = int(r["w"] * s), int(r["h"] * s)
        label = f'{face["dominant_gender"]}, {face["age"]}, {face["dominant_emotion"]}'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("DeepFace Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
cap.release()
cv2.destroyAllWindows()