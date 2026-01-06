
import numpy as np
from torch.utils.data import Dataset
import av
import pandas as pd
from tqdm import tqdm

def echonet_dataset(root_folder: str):
    # derived from ipynb, so the arrays can be created beforehand in mem. w/ np vs using python lists
    # only issue is scalability if the dataset is updated for wtv reason

    # dataset's split is saved within dataset_case 
    NUM_TRAIN_FRAMES = 1315340
    NUM_VAL_FRAMES = 228836
    NUM_TEST_FRAMES = 226460

    VID_DIM = 112

    if (dataset_path := root_folder / "echonet-vit-np_arr.npz").exists():
        print("Found saved EchoNet-Dynamic dataset. Skipping creating views.")
        dataset_case = np.load(dataset_path, mmap_mode="r")

        return dataset_case["TRAIN"]

    video_name_df = pd.read_csv(root_folder / "FileList.csv")

    dataset_case = {
        "TRAIN": [np.zeros((NUM_TRAIN_FRAMES, VID_DIM, VID_DIM), dtype=np.uint8), 0],
        "VAL": [np.zeros((NUM_VAL_FRAMES,     VID_DIM, VID_DIM), dtype=np.uint8), 0],
        "TEST": [np.zeros((NUM_TEST_FRAMES,   VID_DIM, VID_DIM), dtype=np.uint8), 0]
    }
    
    # faster than loc
    split_map = dict(zip(video_name_df["FileName"], video_name_df["Split"]))

    # iterate through each .avi file
    for file_name in tqdm(video_name_df["FileName"]):
        obj = av.open(root_folder / "videos" / (file_name + ".avi"))

        curr_split = split_map[file_name]
        curr_arr, curr_idx = dataset_case[curr_split]

        # iterate through each frame and add it to the np arr
        for frame in obj.decode(video=0):
            frame_arr = frame.to_ndarray(format="gray")

            curr_arr[curr_idx] = frame_arr
            curr_idx += 1
        
        # reupdate stored idx
        dataset_case[curr_split][1] = curr_idx
        obj.close()

    np.savez(root_folder / "echonet-vit-np_arr.npz", **{k: v[0] for k, v in dataset_case.items()})
    print("Finished saving np array.")

    return dataset_case["TRAIN"]