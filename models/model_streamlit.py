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

def process_video(video_path, output_path, model, batch_size=4):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = []
    processed_frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frames.append(frame)

        if len(frames) == batch_size:
            # Process batch
            results = model(frames, classes=[0], conf=0.5)  # 0 is the class index for 'person'

            for i, result in enumerate(results):
                count = len(result.boxes)
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frames[i], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                cv2.putText(frames[i], f'People in frame: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                processed_frames.append(frames[i])

            # Write processed frames to video
            for frame in processed_frames:
                out.write(frame)

            frames = []
            processed_frames = []

    # Process any remaining frames
    if frames:
        results = model(frames, classes=[0], conf=0.5)
        for i, result in enumerate(results):
            count = len(result.boxes)
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frames[i], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(frames[i], f'People in frame: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frames[i])

    video.release()
    out.release()

    return output_path

    # video.release()
    # out.release()
    # cv2.destroyAllWindows()
