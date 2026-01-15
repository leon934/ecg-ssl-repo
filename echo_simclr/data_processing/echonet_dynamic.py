from pathlib import Path

import numpy as np
import av
import pandas as pd
from tqdm import tqdm

def echonet_dataset(root_folder: Path, args: dict) -> np.ndarray:
    dataset_path = root_folder / f"echonet-np_arr-{args.model}.npz"

    if dataset_path.exists():
        print(f"Found saved EchoNet-Dynamic dataset for {args.model} model. Skipping creating views.")
        dataset_dict = np.load(dataset_path, mmap_mode="r", allow_pickle=True)

        return dataset_dict["TRAIN"]

    video_name_df = pd.read_csv(root_folder / "FileList.csv")

    dataset_dict = {
        "TRAIN": [],
        "VAL":   [],
        "TEST":  []
    }
    
    # faster than pandas loc
    split_map = dict(zip(video_name_df["FileName"], video_name_df["Split"]))

    # iterate through each .avi file
    for file_name in tqdm(video_name_df["FileName"]):
        obj = av.open(root_folder / "videos" / (file_name + ".avi"))

        curr_split = split_map[file_name]
        curr_arr = dataset_dict[curr_split]

        video_arr = np.expand_dims(np.stack([frame.to_ndarray(format="gray") for frame in obj.decode(video=0)]), axis=1)
        
        if args.model == "vit": 
            curr_arr.append(video_arr)
        # only want videos >= clip_length, since we rely on sampling a video of length clip_length
        elif args.model == "vivit" and len(video_arr) >= args.clip_length:
            curr_arr.append(video_arr)

    # keep only np arr, not idx
    np.savez(dataset_path, **{k: np.array(v, dtype=object) for k, v in dataset_dict.items()}, )
    print(f"Finished saving EchoNet NumPy array for {args.model} model.")

    return dataset_dict["TRAIN"]