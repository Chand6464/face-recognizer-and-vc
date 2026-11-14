import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)  # 0 = default webcam

frame_count = 0
results = []
SCALE = 0.5  # analyze a smaller frame for speed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze every 15th frame to keep video smooth
    if frame_count % 15 == 0:
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        try:
            results = DeepFace.analyze(
                small,
                actions=["age", "gender", "emotion"],
                detector_backend="opencv",
                enforce_detection=False,
            )
        except Exception as e:
            print("Analysis error:", e)

    # Draw results (scale coordinates back up to the full frame)
    for face in results:
        r = face["region"]
        x = int(r["x"] / SCALE)
        y = int(r["y"] / SCALE)
        w = int(r["w"] / SCALE)
        h = int(r["h"] / SCALE)
        label = f'{face["dominant_gender"]}, {face["age"]}, {face["dominant_emotion"]}'

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("DeepFace Live", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()