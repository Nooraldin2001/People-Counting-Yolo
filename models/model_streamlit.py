import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov9c.pt")

# Define prediction functions
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def process_video(video_path, output_path, roi, model):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mask = np.zeros((height, width), dtype=np.uint8)
    roi_corners = np.array(roi, dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], (255, 255, 255))

    count = 0
    centroids = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame, results = predict_and_detect(model, frame, classes=["person"], conf=0.5)
        people = [box for result in results for box in result.boxes if result.names[int(box.cls[0])] == "person"]

        for person in people:
            x, y = int((person.xyxy[0][0] + person.xyxy[0][2]) / 2), int((person.xyxy[0][1] + person.xyxy[0][3]) / 2)
            if mask[y, x] == 255:
                if not any(np.linalg.norm(np.array([x, y]) - np.array(c)) < 50 for c in centroids):
                    count += 1
                    centroids.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.polylines(frame, [roi_corners], True, (0, 255, 0), 2)
        cv2.putText(frame, f'People in area: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
