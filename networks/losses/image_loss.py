from torch import nn, Tensor

from .build import LOSSES


@LOSSES.register('image')
class ImageLoss(nn.Module):
    def __init__(self, method='mse', masked=False, **kwargs):
        super().__init__()
        self.masked = masked
        self.method = method
        if method == 'l1':
            self.loss_fun = nn.L1Loss(reduction='sum' if masked else 'mean')
        elif method == 'mse':
            self.loss_fun = nn.MSELoss(reduction='sum' if masked else 'mean')
        else:
            raise ValueError(f"method={method} for {self.__class__.__name__} is not supported")

    def forward(self, pred_image: Tensor, gt_image: Tensor, mask: Tensor = None):
        pred_image = pred_image[..., :3]
        if self.masked:
            if mask is None:
                assert gt_image.shape[-1] == 4
                mask = gt_image[..., -1:]
            assert mask.ndim == pred_image.ndim
        else:
            mask = None
        gt_image = gt_image[..., :3]
        if self.masked:
            return self.loss_fun(pred_image * mask, gt_image * mask) / mask.sum().clamp(1e-5)
        else:
            return self.loss_fun(pred_image, gt_image)

    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.method}, masked={self.masked})"
