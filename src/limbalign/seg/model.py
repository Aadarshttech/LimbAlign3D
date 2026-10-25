"""
3D Convolutional Neural Network architectures and sliding window inference for lower-limb bone segmentation.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


def build_limbalign_unet(
    in_channels: int = 1,
    out_channels: int = 5,  # 0: bg, 1: pelvis, 2: femur, 3: tibia, 4: fibula
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Builds a 5-level 3D Residual U-Net for multi-class lower-limb bone segmentation.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm="batch",
    )
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def infer_sliding_window(
    model: nn.Module,
    ct_tensor: torch.Tensor,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Executes Gaussian-blended sliding window inference across a full 3D CT volume.
    Input ct_tensor: shape (1, 1, H, W, D) or (1, H, W, D).
    Returns predicted class label map (H, W, D).
    """
    dev = device or (ct_tensor.device if isinstance(ct_tensor, torch.Tensor) else torch.device("cpu"))
    if ct_tensor.ndim == 4:
        ct_tensor = ct_tensor.unsqueeze(0)

    model.eval()
    ct_tensor = ct_tensor.to(dev)

    raw_logits = sliding_window_inference(
        inputs=ct_tensor,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )

    pred_mask = torch.argmax(raw_logits, dim=1).squeeze(0)
    return pred_mask

