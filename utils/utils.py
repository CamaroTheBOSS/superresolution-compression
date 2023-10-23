import os

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor


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
    for b_prediction, b_label, b_bilinear in zip(img_dict["SR"], img_dict["HR"], img_dict["BILINEAR"]):
        for prediction, label, bilinear in zip(b_prediction, b_label, b_bilinear):
            x1, y1, x2, y2 = get_size_without_zero_padding(label, padding_value=-1)
            prediction = prediction[:, y1:y2, x1:x2]
            label = label[:, y1:y2, x1:x2]
            bilinear = bilinear[:, y1:y2, x1:x2]

            upper_row = torch.cat((prediction, label), dim=2)
            lower_row = torch.cat((bilinear, label), dim=2)
            output = torch.cat((upper_row, lower_row), dim=1)

            output = np.clip((output.detach().permute(1, 2, 0).cpu().numpy() + 1) * 127.5, 0, 255)
            output = output.astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/superresolution-{(4 - len(str(i))) * '0' + str(i)}.png", output)
            i += 1


def get_size_without_zero_padding(image: Tensor, padding_value=0):
    dummy = torch.argwhere(image != padding_value)
    max_x = dummy[:, 2].max()
    min_x = dummy[:, 2].min()
    min_y = dummy[:, 1].min()
    max_y = dummy[:, 1].max()

    return min_x, min_y, max_x, max_y
