# frame extraction
import cv2
import csv
import mediapipe as mp
import os

dataset_csv = "videos.csv"

#folder for spatial dataset


# initialize face detector 
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# get frames
with open(dataset_csv, "r") as f:
    label_reader = csv.DictReader(f)
    for entry in label_reader:
        video_file = entry["video_path"]
        label = entry["label"]

        capture = cv2.VideoCapture(video_file)

        if not capture.isOpened():
            print(f"Could not open video: {video_file}")
            continue

        frame_num = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # run MediaPipe Face Detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            if not results.detections:
                frame_num += 1
                continue

            # get bounding box coordinates
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # skip invalid boxes
            if x2 <= x1 or y2 <= y1:
            	frame_num += 1
            	continue

            # crop to face
            face_frame = frame[y1:y2, x1:x2]

            # skip empty 
            if face_frame.size == 0:
            	frame_num += 1
            	continue

            #resize for CNN
            face_frame = cv2.resize(face_frame, (224, 224))


#feed into CNN