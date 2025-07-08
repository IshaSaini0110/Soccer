# === STEP 1: Import Required Libraries ===
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# === STEP 2: Load YOLOv11 Model ===
model = YOLO('model/best.pt')  # Already downloaded fine-tuned model

# === STEP 3: Initialize DeepSORT Tracker ===
tracker = DeepSort(max_age=1500)  # Keeps tracking IDs for longer after objects leave

# === STEP 4: Load Input Video ===
cap = cv2.VideoCapture('video/15sec_input_720p.mp4')

# Optional: Save Output Video
output = cv2.VideoWriter('output/output_tracked.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         int(cap.get(cv2.CAP_PROP_FPS)),
                         (int(cap.get(3)), int(cap.get(4))))

# === STEP 5: Process Video Frame by Frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # STEP 5.1: Run YOLOv11 on Frame
    results = model(frame)[0]
    detections = []

    # STEP 5.2: Parse Detected Boxes
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        cls = int(cls.item())
        if cls == 0:  # class 0 = player
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], float(conf.item()), 'player'))

    # STEP 5.3: Update DeepSORT Tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # STEP 5.4: Draw Bounding Boxes and Track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # STEP 5.5: Display and Save Frame
    cv2.imshow('Player Tracking', frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === STEP 6: Release Everything ===
cap.release()
output.release()
cv2.destroyAllWindows()
print("✅ Done! Output saved as output_tracked.mp4")
