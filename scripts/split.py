import csv
from sklearn.model_selection import train_test_split

def make_splits(csv_path="videos.csv"):
    """
    Reads videos.csv and returns three lists:
    - train_videos
    - val_videos
    - test_videos

    Each list contains full video paths.
    This ensures the split happens at the VIDEO LEVEL,
    which prevents identity leakage and matches the methodology.
    """

    videos = []
    labels = []

    # Read CSV
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(row["video_path"])
            labels.append(row["label"])

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        videos, labels,
        test_size=0.30,
        stratify=labels,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    return X_train, X_val, X_test
