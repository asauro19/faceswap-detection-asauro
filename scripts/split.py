import json
import os

from torch.utils.data import Subset


def video_info(sample):
    img_path, label = sample
    video_name = os.path.basename(os.path.dirname(img_path))
    label_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    video_ids = os.path.splitext(video_name)[0].split("_")

    if label_name == "original":
        return label_name, video_ids[:1]
    return label_name, video_ids[:2]


def load_split(split_dir, split_name):
    with open(os.path.join(split_dir, f"{split_name}.json"), "r") as f:
        pairs = json.load(f)

    originals = set()
    manipulated = set()

    for a, b in pairs:
        a = str(a).zfill(3)
        b = str(b).zfill(3)

        originals.add(a)
        originals.add(b)
        manipulated.add((a, b))
        manipulated.add((b, a))

    return originals, manipulated


def split_indices(dataset, split_dir, split_name):
    originals, manipulated = load_split(split_dir, split_name)
    indices = []

    for i, sample in enumerate(dataset.samples):
        label_name, video_ids = video_info(sample)

        if label_name == "original" and video_ids[0] in originals:
            indices.append(i)
        elif label_name == "manipulated" and tuple(video_ids[:2]) in manipulated:
            indices.append(i)

    return indices


def make_faceforensics_splits(dataset, split_dir):
    indices = {
        "train": split_indices(dataset, split_dir, "train"),
        "val": split_indices(dataset, split_dir, "val"),
        "test": split_indices(dataset, split_dir, "test"),
    }

    for split_name, split_list in indices.items():
        if len(split_list) == 0:
            raise ValueError(
                f"No samples found for {split_name} split. "
                "Check that extracted folders use FaceForensics names like "
                "original/071 and manipulated/071_054."
            )

    # Make sure no source video id appears in more than one split.
    seen = {}
    for split_name, split_list in indices.items():
        for i in split_list:
            _, video_ids = video_info(dataset.samples[i])
            for video_id in video_ids:
                if video_id in seen and seen[video_id] != split_name:
                    raise ValueError(
                        f"Video id {video_id} appears in both {seen[video_id]} and {split_name}"
                    )
                seen[video_id] = split_name

    return (
        Subset(dataset, indices["train"]),
        Subset(dataset, indices["val"]),
        Subset(dataset, indices["test"]),
    )
