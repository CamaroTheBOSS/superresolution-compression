import os
import uuid

import cv2
import numpy as np
import yaml


def init_training_directory(output_root: str, run_name: str) -> str:
    run_path = os.path.join(output_root, run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    return run_path


def save_args(training_directory, **kwargs):
    with open(os.path.join(training_directory, "arguments.yaml"), 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=True)


def save_results(img_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 0
    for batch_list in img_dict["SR"]:
        for frame in batch_list:
            img = frame.detach().permute(1, 2, 0).cpu().numpy() * 255.
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/superresolution-{(4 - len(str(i))) * '0' + str(i)}.png", img)
            i += 1
