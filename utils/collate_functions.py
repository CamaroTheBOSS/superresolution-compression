import numpy as np
import torch
from torch.nn.functional import pad


def get_max_size_from_inputs(inputs):
    max_size_lr, max_size_hr, min_nframe = (0, 0), (0, 0), None
    for low_res, high_res in inputs:
        if len(low_res.shape) == 4:
            min_nframe = min(min_nframe, low_res.shape[0]) if min_nframe is not None else low_res.shape[0]

        lr_shape = low_res.shape[-2:]
        hr_shape = high_res.shape[-2:]
        max_size_lr = max(max_size_lr[0], lr_shape[0]), max(max_size_lr[1], lr_shape[1])
        max_size_hr = max(max_size_hr[0], hr_shape[0]), max(max_size_hr[1], hr_shape[1])
    return max_size_lr, max_size_hr, min_nframe


def align_images_with_pad(inputs):
    max_size_lr, max_size_hr, min_nframe = get_max_size_from_inputs(inputs)
    for e, (low_res, high_res) in enumerate(inputs):
        lr_shape = low_res.shape[-2:]
        hr_shape = high_res.shape[-2:]

        pad_updown, pad_side = (max_size_lr[0] - lr_shape[0]), (max_size_lr[1] - lr_shape[1])
        low_res_padded = pad(low_res, (0, pad_side, 0, pad_updown), value=-1)

        pad_updown, pad_side = (max_size_hr[0] - hr_shape[0]), (max_size_hr[1] - hr_shape[1])
        high_res_padded = pad(high_res, (0, pad_side, 0, pad_updown), value=-1)

        if min_nframe is not None:
            start_frame = np.random.randint(0, low_res_padded.shape[0] - min_nframe + 1)
            low_res_padded = low_res_padded[start_frame:start_frame + min_nframe]
            high_res_padded = high_res_padded[start_frame:start_frame + min_nframe]
        inputs[e] = (low_res_padded, high_res_padded)

    return inputs


def collate_SR(inputs) -> dict:
    aligned_inputs = align_images_with_pad(inputs)
    batch = {"LR": torch.stack([i[0] for i in aligned_inputs], dim=0),
             "HR": torch.stack([i[1] for i in aligned_inputs], dim=0)}
    return batch
