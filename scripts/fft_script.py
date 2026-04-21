import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "rgb_faces"
OUTPUT_DIR = "fft_faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(INPUT_DIR):  # original / manipulated
    label_path = os.path.join(INPUT_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for video_name in os.listdir(label_path):  # 000 / 000_003
        video_path = os.path.join(label_path, video_name)
        if not os.path.isdir(video_path):
            continue

        out_dir = os.path.join(OUTPUT_DIR, label, video_name)
        os.makedirs(out_dir, exist_ok=True)

        images = sorted(os.listdir(video_path))


        pbar = tqdm(images, desc=f"FFT {label}/{video_name}")
    

        for img_file in pbar:
            img_path = os.path.join(video_path, img_file)

            # load grayscale image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue


            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)


            # normalize to 0–255
            magnitude = magnitude / magnitude.max() * 255
            magnitude = magnitude.astype(np.uint8)

            # save as fface_XXXX.jpg
            out_file = img_file.replace("sface", "fface")
            out_path = os.path.join(out_dir, out_file)
            cv2.imwrite(out_path, magnitude)
