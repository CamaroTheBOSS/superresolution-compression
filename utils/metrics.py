import torch
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from utils.utils import get_size_without_zero_padding


def psnr(input_images: Tensor, target_images: Tensor) -> Tensor:
    if len(input_images.shape) == 4:
        return _psnr_images(input_images, target_images)
    elif len(input_images.shape) == 5:
        return _psnr_videos(input_images, target_images)

    raise NotImplementedError("Input tensors should be 4D or 5D")


def ssim(input_images: Tensor, target_images: Tensor) -> Tensor:
    if len(input_images.shape) == 4:
        return _ssim_images(input_images, target_images)
    elif len(input_images.shape) == 5:
        return _ssim_videos(input_images, target_images)

    raise NotImplementedError("Input tensors should be 4D or 5D")


def _psnr_images(input_images: Tensor, target_images: Tensor) -> Tensor:
    """
    Function, which calculates PSNR for batch of images
    :param input_images: Tensor with shape B,C,H,W low resolution image
    :param target_images: Tensor with shape B,C,H,W high resolution image
    :return: sum of PSNR values in batch
    """
    if not len(input_images.shape) == 4:
        raise NotImplementedError("Input tensors should have 4D shape B,C,H,W")

    psnr_values = []
    for i in range(target_images.size()[0]):
        x1, y1, x2, y2 = get_size_without_zero_padding(target_images[i], padding_value=-1)
        psnr_values.append(peak_signal_noise_ratio(input_images[i][:, y1:y2, x1:x2],
                                                   target_images[i][:, y1:y2, x1:x2], 1.0))
    return torch.sum(torch.stack(psnr_values))


def _psnr_videos(input_videos: Tensor, target_videos: Tensor) -> Tensor:
    """
    Function, which calculates PSNR for batch of videos
    :param input_videos: Tensor with shape B,N,C,H,W low resolution video
    :param target_videos: Tensor with shape B,N,C,H,W high resolution video
    :return: sum of PSNR values in batch
    """
    if not len(input_videos.shape) == 5:
        raise NotImplementedError("Input tensors should have 5D shape B,N,C,H,W")

    psnr_values = []
    for i in range(target_videos.size()[0]):
        x1, y1, x2, y2 = get_size_without_zero_padding(target_videos[i][0], padding_value=-1)
        psnr_values.append(peak_signal_noise_ratio(input_videos[i][:, :, y1:y2, x1:x2],
                                                   target_videos[i][:, :, y1:y2, x1:x2], 1.0))
    return torch.sum(torch.stack(psnr_values))


def _ssim_images(input_images: Tensor, target_images: Tensor) -> Tensor:
    """
       Function, which calculates SSIM for batch of images
       :param input_images: Tensor with shape B,C,H,W low resolution image
       :param target_images: Tensor with shape B,C,H,W high resolution image
       :return: sum of SSIM values in batch
       """
    if not len(input_images.shape) == 4:
        raise NotImplementedError("Input tensors should have 4D shape B,C,H,W")

    ssim_values = []
    for i in range(target_images.size()[0]):
        x1, y1, x2, y2 = get_size_without_zero_padding(target_images[i], padding_value=-1)
        ssim_values.append(structural_similarity_index_measure(input_images[i][:, y1:y2, x1:x2].unsqueeze(0),
                                                               target_images[i][:, y1:y2, x1:x2].unsqueeze(0)))
    return torch.sum(torch.stack(ssim_values))


def _ssim_videos(input_videos: Tensor, target_videos: Tensor) -> Tensor:
    """
       Function, which calculates SSIM for batch of images
       :param input_videos: Tensor with shape B,N,C,H,W low resolution video
       :param target_videos: Tensor with shape B,N,C,H,W high resolution video
       :return: sum of SSIM values in batch
       """
    if not len(input_videos.shape) == 5:
        raise NotImplementedError("Input tensors should have 5D shape B,N,C,H,W")

    ssim_values = []
    for i in range(target_videos.size()[0]):
        x1, y1, x2, y2 = get_size_without_zero_padding(target_videos[i][0], padding_value=-1)
        ssim_values.append(structural_similarity_index_measure(input_videos[i][:, :, y1:y2, x1:x2],
                                                               target_videos[i][:, :, y1:y2, x1:x2]))
    return torch.sum(torch.stack(ssim_values))