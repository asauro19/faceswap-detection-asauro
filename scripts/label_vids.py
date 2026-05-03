import os
import csv

originals_path = "FaceSwapData/original_sequences/youtube/c23/videos"
manipulated_path = "FaceSwapData/manipulated_sequences/FaceSwap/c23/videos"

#list original videos 
original_vids = [v for v in os.listdir(originals_path) if v.endswith(".mp4")]

#list mainpulated videos 
manipulated_vids = [f for f in os.listdir(manipulated_path) if f.endswith(".mp4")]

#need to label data 
csv_rows = []

for v in original_vids:
    path = os.path.join(originals_path, v)
    csv_rows.append([path, "original"])

for f in manipulated_vids:
    path = os.path.join(manipulated_path, f)
    csv_rows.append([path, "manipulated"])

#create csv file with videos labeled correctly
with open("videos.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "label"])
    writer.writerows(csv_rows)

#check
print("Original:", len(original_vids))
print("Manipulated:", len(manipulated_vids))
