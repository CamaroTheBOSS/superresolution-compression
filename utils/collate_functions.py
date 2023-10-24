import torch
from torch.nn.functional import pad


def get_max_size_from_inputs(inputs):
    max_size_lr, max_size_hr = (0, 0), (0, 0)
    for low_res, high_res in inputs:
        lr_shape = low_res.shape[-2:]
        hr_shape = high_res.shape[-2:]
        max_size_lr = max(max_size_lr[0], lr_shape[0]), max(max_size_lr[1], lr_shape[1])
        max_size_hr = max(max_size_hr[0], hr_shape[0]), max(max_size_hr[1], hr_shape[1])
    return max_size_lr, max_size_hr


def align_images_with_pad(inputs):
    max_size_lr, max_size_hr = get_max_size_from_inputs(inputs)
    for e, (low_res, high_res) in enumerate(inputs):
        lr_shape = low_res.shape[-2:]
        hr_shape = high_res.shape[-2:]

        pad_updown, pad_side = (max_size_lr[0] - lr_shape[0]), (max_size_lr[1] - lr_shape[1])
        low_res_padded = pad(low_res, (0, pad_side, 0, pad_updown), value=-1)

        pad_updown, pad_side = (max_size_hr[0] - hr_shape[0]), (max_size_hr[1] - hr_shape[1])
        high_res_padded = pad(high_res, (0, pad_side, 0, pad_updown), value=-1)

        inputs[e] = (low_res_padded, high_res_padded)

    return inputs


def collate_image_superresolution(inputs) -> dict:
    lr_shapes = torch.stack([torch.Tensor(tuple(i[0].shape[-2:])) for i in inputs])
    hr_shapes = torch.stack([torch.Tensor(tuple(i[0].shape[-2:])) for i in inputs])
    aligned_inputs = align_images_with_pad(inputs)
    batch = {"LR": torch.stack([i[0] for i in aligned_inputs], dim=0),
             "HR": torch.stack([i[1] for i in aligned_inputs], dim=0),
             "LR_shapes": lr_shapes,
             "HR_shapes": hr_shapes}
    return batch
