import torch
from torch import nn, SymInt
from torchvision.transforms import GaussianBlur
from typing import Literal, Callable, Sequence, Union

from .conv import AverageConvnd, GaussianConvnd

class SSIMLoss(nn.Module):
    r"""
    Creates a criterion that measures the SSIM (structural similarity index measure) between
    each element in the input and target.

    Supports 1d, 2d, 3d input.

    Attributes:
        kernel_size: The size of the sliding window. Must be an int, or a shape with 1, 2 or 3 dimensions.
        *,
        kernel_type: Type of kernel ("avg" or "gauss") or a Custom callable object.
        reduction: Reduction method ("mean", "sum", or "none").
        data_range: Dynamic range of input tensors.
        k1,k2: Stabilization constants for SSIM calculation.
    """

    conv: Callable[..., torch.Tensor]
    data_range: torch.Tensor

    def __init__(
            self,
            kernel_size: int | Sequence[Union[int, SymInt]],
            *,
            kernel: Literal["avg", "gauss"] | Callable[[torch.Tensor], torch.Tensor] = "gauss",
            reduction: Literal["mean", "sum", "none"] = "mean",
            data_range: float = 1.0,
            k1: float = 0.01, k2: float = 0.03
        ):
        super().__init__()
        self.kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
        self.kernel = kernel
        self.reduction = reduction
        self.register_buffer("data_range", torch.as_tensor(data_range))
        self.k1, self.k2 = k1, k2

        # unbiased estimate
        npts = torch.as_tensor(self.kernel_size).numel()
        self.cov_norm = npts / (npts - 1)

        ndim = len(self.kernel_size)
        if self.kernel == "avg":
            self.conv = AverageConvnd(self.kernel_size)
        elif self.kernel == "gauss":
            if ndim == 2:
                self.conv = GaussianBlur(self.kernel_size, sigma = 1.5)
            else:
                self.conv = GaussianConvnd(self.kernel_size, sigma = 1.5)
        elif callable(self.kernel):
            self.conv = self.kernel
        else:
            raise ValueError("`kernel` only supports 'avg', 'gauss' or Callable object.")

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            batch_data_range: torch.Tensor = None,
        ):
        if batch_data_range is None:
            data_range = self.data_range
        else:
            data_range = batch_data_range

        if data_range.ndim > 0 and input.ndim >= 3:
            if input.size(0) != data_range.size(0):
                raise ValueError(f"`input` and `batch_data_range` must have the same batchsize. Got shape `input` {input.shape}, `data_range` {data_range.shape}")
            data_range = data_range.view(input.size(0),*([1] * (input.ndim - 1)))

        if input.ndim < 4:
            input = input.view(
                *(1,) * (4 - input.ndim), *input.shape,
            )
        if target.ndim < 4:
            target = target.view(
                *(1,) * (4 - target.ndim), *target.shape,
            )

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = self.conv(input)
        uy = self.conv(target)
        uxx = self.conv(input**2)
        uyy = self.conv(target**2)
        uxy = self.conv(input * target)

        vx = self.cov_norm * (uxx - ux**2)
        vy = self.cov_norm * (uyy - uy**2)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux**2 + uy**2 + C1
        B2 = vx + vy + C2
        
        ssim_map = (A1 * A2) / (B1 * B2)
        
        if self.reduction == "mean":
            return 1 - ssim_map.mean()
        elif self.reduction == "sum":
            return ssim_map.size(0) - ssim_map.mean(tuple(range(1, ssim_map.ndim))).sum()
        else:
            return 1 - ssim_map
