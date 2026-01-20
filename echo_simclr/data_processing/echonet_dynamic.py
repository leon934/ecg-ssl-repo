from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm

def echonet_dataset(root_folder: Path, args: dict) -> np.ndarray:
    dataset_path = root_folder / f"echonet-np_arr-{args.model}.npz"

    if dataset_path.exists():
        print(f"Found saved EchoNet-Dynamic dataset for {args.model} model. Skipping creating views.")
        dataset_dict = np.load(dataset_path, mmap_mode="r", allow_pickle=True)

        return dataset_dict

    video_name_df = pd.read_csv(root_folder / "FileList.csv",)

    splits = ("X", "TRAIN", "VAL", "TEST")
    dataset_dict, ef_dict, es_dict, edv_dict = (
        {s: [] for s in splits} for _ in range(4)
    )

    # iterate through each .avi file
    for row in tqdm(video_name_df.itertuples(), total=len(video_name_df)):
        obj = av.open(root_folder / "videos" / (row.FileName + ".avi"))
        curr_split = row.Split

        video_arr = np.expand_dims(np.stack([frame.to_ndarray(format="gray") for frame in obj.decode(video=0)]), axis=1)
        
        if args.model == "vit": 
            dataset_dict[curr_split].append(video_arr)

            ef_dict[curr_split].append(row.EF)
            es_dict[curr_split].append(row.ESV)
            edv_dict[curr_split].append(row.EDV)
        # only want videos >= clip_length, since we rely on sampling a video of length clip_length
        elif args.model == "vivit" and len(video_arr) >= args.clip_length:
            dataset_dict[curr_split].append(video_arr)

            ef_dict[curr_split].append(row.EF)
            es_dict[curr_split].append(row.ESV)
            edv_dict[curr_split].append(row.EDV)

    save_arr = {}

    for split in ("TRAIN", "TEST", "VAL"):
        save_arr[f"X_{split}"] = np.array(dataset_dict[split], dtype=object)
        save_arr[f"EF_{split}"] = np.array(ef_dict[split], dtype=object)
        save_arr[f"ESV_{split}"] = np.array(es_dict[split], dtype=object)
        save_arr[f"EDV_{split}"] = np.array(edv_dict[split], dtype=object)

    np.savez(dataset_path, **{k: np.array(v, dtype=object) for k, v in dataset_dict.items()})
    print(f"Finished saving EchoNet NumPy array for {args.model} model.")

    return dataset_dict