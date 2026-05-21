from pathlib import Path
import h5py
import numpy as np
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "data" / "raw"
folders = [
    BASE_DIR / "Intra" / "train",
    BASE_DIR / "Intra" / "test",
    BASE_DIR / "Cross" / "train",
    BASE_DIR / "Cross" / "test1",
    BASE_DIR / "Cross" / "test2",
    BASE_DIR / "Cross" / "test3",
]


def get_label(filename):
    name = Path(filename).stem

    if name.startswith("rest"):
        return "rest"
    if name.startswith("task_story_math"):
        return "story_math"
    if name.startswith("task_working_memory"):
        return "working_memory"
    if name.startswith("task_motor"):
        return "motor"

    return "unknown"


for folder in folders:
    print("\n" + "=" * 60)
    print(folder)

    files = sorted(folder.glob("*.h5"))
    print("Number of files:", len(files))

    labels = [get_label(f.name) for f in files]
    print("Labels:", Counter(labels))

    if len(files) > 0:
        example_file = files[0]

        with h5py.File(example_file, "r") as f:
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][()]

        print("Example file:", example_file.name)
        print("Dataset name:", dataset_name)
        print("Shape:", data.shape)
        print("Min:", np.min(data))
        print("Max:", np.max(data))
        print("Mean:", np.mean(data))
        print("Std:", np.std(data))