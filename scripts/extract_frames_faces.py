import cv2
import csv
import mediapipe as mp
import os
from tqdm import tqdm
from split import make_splits

# Load train/val/test video lists
train_videos, val_videos, test_videos = make_splits("../videos.csv")

# MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def extract_faces(video_list, split_name):
    """
    Extract faces ONLY from videos in the given split.
    Saves to spatial_faces/<split_name>/<label>/<video_name>/
    """
    for video_file in video_list:
        # Determine label from path
        label = "original" if "original_sequences" in video_file else "manipulated"

        # Open video
        capture = cv2.VideoCapture(video_file)
        if not capture.isOpened():
            print(f"Could not open video: {video_file}")
            continue

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc=f"[{split_name}] {os.path.basename(video_file)}")

        frame_num = 0
        video_name = os.path.splitext(os.path.basename(video_file))[0]

        # Output folder
        out_dir = os.path.join("spatial_faces", split_name, label, video_name)
        os.makedirs(out_dir, exist_ok=True)

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # Run MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box

                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 > x1 and y2 > y1:
                    face_frame = frame[y1:y2, x1:x2]

                    if face_frame.size > 0:
                        out_path = os.path.join(out_dir, f"sface_{frame_num:04d}.jpg")
                        cv2.imwrite(out_path, face_frame)

            frame_num += 1
            pbar.update(1)

        pbar.close()
        capture.release()


# Run extraction for each split
extract_faces(train_videos, "train")
extract_faces(val_videos, "val")
extract_faces(test_videos, "test")

print("Spatial face extraction complete.")
