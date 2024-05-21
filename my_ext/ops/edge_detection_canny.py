import os
import math

import torch
import torch.nn.functional as F

gaussian_filter = None


def GaussianBlur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 0):
    global gaussian_filter
    padding = kernel_size // 2
    if gaussian_filter is None or gaussian_filter.shape[-1] != kernel_size:
        if sigma <= 0:
            sigma = (kernel_size * 0.5 - 1.) * 0.3 + 0.8
        assert kernel_size > 0 and kernel_size % 2 == 1
        dx = torch.arange(kernel_size, dtype=torch.float).unsqueeze(0) - padding
        dy = torch.arange(kernel_size, dtype=torch.float).unsqueeze(1) - padding
        gaussian_filter = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)) / (2 * sigma ** 2 * math.pi)
        gaussian_filter /= gaussian_filter.sum()
        gaussian_filter = gaussian_filter.to(img.device)
        assert img.ndim == 4
    img = F.pad(img, (padding, padding, padding, padding), mode='replicate')
    return F.conv2d(img, gaussian_filter[None, None, :, :], groups=img.shape[1])


def NMS(gradients: torch.Tensor, direction: torch.Tensor):
    B, H, W = gradients.shape
    g = gradients.unsqueeze(1)

    # assert -math.pi <= direction.min().item() and direction.max().item() <= math.pi, \
    #     f"{direction.min()} {direction.max()}"
    pos = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))[::-1], dim=-1).to(gradients.device)
    pos = pos.unsqueeze(0).repeat((B, 1, 1, 1)).float()
    weight = torch.tan(direction)
    mask = torch.logical_and(direction.ge(-math.pi / 4), direction.lt(math.pi / 4)).float()
    pos[:, :, :, 0] += mask
    pos[:, :, :, 1] += weight * mask
    mask = torch.logical_and(direction.ge(math.pi / 4), direction.lt(3 * math.pi / 4)).float()
    pos[:, :, :, 0] += weight.reciprocal() * mask
    pos[:, :, :, 1] += mask
    mask = torch.logical_or(direction.ge(3 * math.pi / 4), direction.lt(-3 * math.pi / 4)).float()
    pos[:, :, :, 0] -= mask
    pos[:, :, :, 1] -= weight * mask
    mask = torch.logical_and(direction.ge(-3 * math.pi / 4), direction.lt(- math.pi / 4)).float()
    pos[:, :, :, 0] -= weight.reciprocal() * mask
    pos[:, :, :, 1] -= mask
    pos = pos / (H - 1) * 2 - 1
    g1 = F.grid_sample(g, pos, mode='bilinear', align_corners=True)

    direction = direction + math.pi  # [0, 2*pi]
    weight = torch.tan(direction)
    # assert 0 <= direction.min().item() and direction.max().item() <= math.pi * 2
    pos = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))[::-1], dim=-1).to(gradients.device)
    pos = pos.unsqueeze(0).repeat((B, 1, 1, 1)).float()
    mask = torch.logical_or(direction.ge(7 * math.pi / 4), direction.lt(math.pi / 4)).float()
    pos[:, :, :, 0] += mask
    pos[:, :, :, 1] += weight * mask
    mask = torch.logical_and(direction.ge(math.pi / 4), direction.lt(3 * math.pi / 4)).float()
    pos[:, :, :, 0] += weight.reciprocal() * mask
    pos[:, :, :, 1] += mask
    mask = torch.logical_and(direction.ge(3 * math.pi / 4), direction.lt(5 * math.pi / 4)).float()
    pos[:, :, :, 0] -= mask
    pos[:, :, :, 1] -= weight * mask
    mask = torch.logical_and(direction.ge(5 * math.pi / 4), direction.lt(7 * math.pi / 4)).float()
    pos[:, :, :, 0] -= weight.reciprocal() * mask
    pos[:, :, :, 1] -= mask
    pos = pos / (H - 1) * 2 - 1
    g2 = F.grid_sample(g, pos, mode='bilinear', align_corners=False)

    mask = torch.logical_and(g1.le(g), g2.le(g)).float().squeeze(1)
    return gradients * mask


def canny(gray: torch.Tensor):
    G_canny = gray.new_tensor([
        [[[-1., 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        [[[-1., -2, -1],
          [0, 0, 0],
          [1, 2, 1]]],
    ])

    gray = GaussianBlur(gray, kernel_size=3, sigma=0)  # 高斯模糊

    gradient = F.conv2d(gray, G_canny, padding=1)
    direction = torch.atan2(gradient[:, 1], gradient[:, 0])
    gradient = gradient.pow(2).sum(dim=1).sqrt()
    # mi, mx = gradient.min(), gradient.max()
    # print(mi, mx, gradient.shape)
    # gradient[:, (0, -1), :] = 0
    # gradient[:, :, (0, -1)] = 0
    # _temp = (gradient - mi) / (mx - mi) * 255
    # plt.imshow(_temp.to(torch.uint8)[0].cpu().numpy(), cmap='gray')
    # plt.show()
    gradient = NMS(gradient, direction)
    gradient[:, (0, -1), :] = 0
    gradient[:, :, (0, -1)] = 0

    return gradient


# dog = GaussianBlur(gray, 5, 2) - GaussianBlur(gray, 3, 1)
def test():
    import numpy as np
    from PIL import Image
    from pylab import mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    mpl.rcParams['font.size'] = 18

    torch.set_printoptions(2, sci_mode=False, linewidth=200)
    img = Image.open(os.path.expanduser('~/bcnn/docs/pics/cat.jpg')).convert('RGB')
    img = np.array(img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    images = torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0).float()
    gray = images * images.new_tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1)
    gray = gray.sum(dim=1, keepdim=True)
    gradient = canny(gray)
    mi, mx = gradient.min(), gradient.max()
    print(mi, mx, gradient.shape)
    _temp = (gradient - mi) / (mx - mi) * 255
    plt.imshow(_temp.to(torch.uint8)[0].cpu().numpy(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    test()
