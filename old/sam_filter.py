# custom_dataset/filter/sam_filt.py
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import os
from tqdm import tqdm
import os.path
import sys
from typing import Any, Callable, List, Optional, Tuple
import tqdm
from PIL import Image
from torch.utils.data import Dataset
import pickle
from torchvision import transforms
from matplotlib import pyplot as plt
import math
import argparse
import socket
import time
from sam import SamDataset
from urban_filter import Urban_filter

class Args:
    check = False
    mode = "clip_logit"  # Choose from the supported modes
    start_idx = 0
    end_idx = int(9e10)

args = Args()

caption_folder_path = "/content/captions"
image_folder_path = "/content/images"
id_dict_dir = "/content/id_dict"
filt_dir = "/content/filt_result"

os.makedirs(filt_dir, exist_ok=True)

@torch.no_grad()
def main(args):
    filter = Urban_filter()
    if args.mode in ["caption_filt", "gather_result"]:
        filter.clip_filter = None
        torch.cuda.empty_cache()

    def collate_fn(examples):
        ret = {}
        if "image" in examples[0]:
            ret["images"] = [example["image"] for example in examples]
        if "text" in examples[0]:
            ret["text"] = [example["text"] for example in examples]
        ret["ids"] = [example["id"] for example in examples]
        return ret

    error_files = []
    all_remain_ids = []
    all_remain_ids_train = []
    all_remain_ids_val = []
    all_filtered_id_num = 0
    remain_feat_num = 0
    remain_caption_num = 0
    filter_feat_num = 0
    filter_caption_num = 0

    for idx, file in tqdm(enumerate(sorted(os.listdir(id_dict_dir)))):
        if idx < args.start_idx or idx >= args.end_idx:
            continue
        if not file.endswith(".pickle") or file.startswith("all"):
            continue

        print("=====================================")
        print(file, flush=True)
        save_dir = os.path.join(filt_dir, file.replace("_id_dict.pickle", ""))
        os.makedirs(save_dir, exist_ok=True)

        id_dict_file = os.path.join(id_dict_dir, file)
        with open(id_dict_file, 'rb') as f:
            id_dict = pickle.load(f)

        ids = list(id_dict.keys())
        dataset = SamDataset(image_folder_path, caption_folder_path, id_file=ids, id_dict_file=id_dict_file)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

        clip_logits_file = os.path.join(save_dir, "clip_logits_result.pickle")

        if args.mode == "clip_logit":
            skip = os.path.exists(clip_logits_file)
            if skip:
                try:
                    with open(clip_logits_file, 'rb') as f:
                        clip_logits = pickle.load(f)
                    if args.check and clip_logits == "":
                        skip = False
                except:
                    skip = False

            if not skip:
                with open(clip_logits_file, 'wb') as f:
                    pickle.dump("", f)
                try:
                    clip_logits = filter.clip_logit(dataloader)
                except Exception as e:
                    print(f"Error in clip_logit {file}: {e}", flush=True)
                    continue
                with open(clip_logits_file, 'wb') as f:
                    pickle.dump(clip_logits, f)
                print(f"clip_logits_result saved to {clip_logits_file}", flush=True)
            else:
                print(f"skip {clip_logits_file}", flush=True)

    print("Done", flush=True)
    for file in error_files:
        print(f"Error in file: {file}", flush=True)

main(args)
