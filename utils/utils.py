import os
import shutil
from typing import Tuple, Union, List

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor


def init_training_directory(output_root: str, run_name: str) -> str:
    run_path = os.path.join(output_root, run_name)
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path)
    return run_path


def save_args(training_directory, **kwargs) -> None:
    with open(os.path.join(training_directory, "arguments.yaml"), 'w') as f:
        yaml.safe_dump(kwargs, f, sort_keys=True)


def save_results(img_dict: dict, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(img_dict["SR"][0].shape) == 4:
        return _save_results_images(img_dict, output_dir)
    elif len(img_dict["SR"][0].shape) == 5:
        return _save_results_videos(img_dict, output_dir)

    raise NotImplementedError("img_dict['SR'] should contain tensors with shape 4D or 5D")


def get_size_without_zero_padding(image: Tensor, padding_value=0) -> Tuple[int, int, int, int]:
    dummy = torch.argwhere(image != padding_value)
    max_x = dummy[:, 2].max()
    min_x = dummy[:, 2].min()
    min_y = dummy[:, 1].min()
    max_y = dummy[:, 1].max()

    return min_x, min_y, max_x, max_y


def interpolate(input_tensor: Tensor, size: Union[List, Tuple[int, int]], mode: str = "bilinear",
                align_corners: bool = False) -> Tensor:
    if len(input_tensor.shape) == 4:
        return torch.nn.functional.interpolate(input_tensor, size=size, mode=mode, align_corners=align_corners)
    elif len(input_tensor.shape) == 5:
        return torch.vmap(
                torch.nn.functional.interpolate, in_dims=(1,), out_dims=(1,)
            )(input_tensor.transpose(1, 2), size=size, mode=mode, align_corners=align_corners)
    raise NotImplementedError("Input tensors should be 4D or 5D")


def _save_results_images(img_dict: dict, output_dir: str) -> None:
    for b_prediction, b_label, b_bilinear in zip(img_dict["SR"], img_dict["HR"], img_dict["BILINEAR"]):
        for image_idx, (prediction, label, bilinear) in enumerate(zip(b_prediction, b_label, b_bilinear)):
            x1, y1, x2, y2 = get_size_without_zero_padding(label, padding_value=-1)
            prediction = prediction[:, y1:y2, x1:x2]
            label = label[:, y1:y2, x1:x2]
            bilinear = bilinear[:, y1:y2, x1:x2]

            upper_row = torch.cat((prediction, label), dim=2)
            lower_row = torch.cat((bilinear, torch.zeros_like(label)), dim=2)
            output = torch.cat((upper_row, lower_row), dim=1)

            output = np.clip((output.detach().permute(1, 2, 0).cpu().numpy()) * 255., 0, 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/{(4 - len(str(image_idx))) * '0' + str(image_idx)}.png", output)


def _save_results_videos(img_dict: dict, output_dir: str) -> None:
    for b_idx, (b_pred, b_label, b_bilinear) in enumerate(zip(img_dict["SR"], img_dict["HR"], img_dict["BILINEAR"])):
        for v_idx, (v_prediction, v_label, v_bilinear) in enumerate(zip(b_pred, b_label, b_bilinear)):
            x1, y1, x2, y2 = get_size_without_zero_padding(v_label[0], padding_value=-1)
            v_prediction = v_prediction[:, :, y1:y2, x1:x2]
            v_label = v_label[:, :, y1:y2, x1:x2]
            v_bilinear = v_bilinear[:, :, y1:y2, x1:x2]

            upper_row = torch.cat((v_prediction, v_label), dim=3)
            lower_row = torch.cat((v_bilinear, torch.zeros_like(v_label)), dim=3)
            out_fr = torch.cat((upper_row, lower_row), dim=2)
            out_fr = np.clip((out_fr.detach().permute(0, 2, 3, 1).cpu().numpy()) * 255., 0, 255).astype(np.uint8)

            subfolder_idx = b_idx * len(b_pred) + v_idx
            subfolder = os.path.join(output_dir, (4 - len(str(subfolder_idx))) * '0' + str(subfolder_idx))
            if os.path.exists(subfolder):
                shutil.rmtree(subfolder)
            os.makedirs(subfolder)
            for frame_idx, frame in enumerate(out_fr):
                out_fr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{subfolder}/{(4 - len(str(frame_idx))) * '0' + str(frame_idx)}.png", out_fr)
