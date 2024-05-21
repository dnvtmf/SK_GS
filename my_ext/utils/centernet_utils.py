from typing import Union

import numpy as np
import torch
from torch import nn, Tensor


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = hmax.eq(heat).float()
    return heat * keep


def topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).long()
    topk_inds = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat: Tensor, ind: Tensor, mask=None) -> Tensor:
    """
    Args:
        feat: shape: (B, HW, C)
        ind: shape: (B, N)
        mask: optial
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_features(feat: Tensor, ind: Tensor) -> Tensor:
    """
    Args:
        feat: shape: (B, C, H, W)
        ind: shape: (B, N)
    return: shape: (B, N, C)
    """
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
    ind = ind.unsqueeze(2).expand(B, -1, C)
    feat = feat.gather(1, ind)
    return feat


def _gather_features(feat: Tensor, x: Tensor, y: Tensor = None) -> Tensor:
    """ faster than transpose_and_gather_features for forward
    Args:
        feat: shape: (B, C, H, W)
        x: shape: (B, N) or (B, N, 2)
        y: shape: (B, N) or None
    return: shape: (B, N, C)
    """
    B, C, H, W = feat.shape
    if x.ndim == 3:
        x, y = x[..., 0], x[..., 1]
    b = torch.arange(B, device=feat.device, dtype=torch.long).view(B, 1).expand_as(x)
    return feat[b, :, y, x]


def sigmoid_(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1.):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap: Union[np.ndarray, Tensor], center, radius: int, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        if isinstance(heatmap, np.ndarray):
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        else:
            masked_gaussian = torch.from_numpy(masked_gaussian).to(masked_heatmap)
            torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape((-1, 1, 1))
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # _TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
        np.maximum(heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def test_speed_for_gather_features():
    import my_ext as ext
    import torch.profiler
    print()
    ext.utils.set_printoptions()
    K = 100
    H = W = 512 // 4
    f = torch.randn(32, 10, H, W).cuda()  # type:torch.Tensor
    x = torch.randint(0, W, (32, K)).cuda()
    y = torch.randint(0, H, (32, K)).cuda()
    ind = y * W + x

    o1 = transpose_and_gather_features(f, ind)
    o2 = _gather_features(f, x, y)
    assert (o1 - o2).abs().max() == 0.0
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as p:
        for _ in range(10):
            transpose_and_gather_features(f, ind)
            # gather_features(f, x, y)

    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as p:
        for _ in range(10):
            # transpose_and_gather_features(f, ind)
            _gather_features(f, x, y)
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=-1))

    import torch.utils.benchmark as benchmark

    t0 = benchmark.Timer(
        stmt='transpose_and_gather_features(f, ind)',
        setup='from __main__ import transpose_and_gather_features',
        globals={'f': f, 'ind': ind})

    t1 = benchmark.Timer(
        stmt='_gather_features(f, x, y)',
        setup='from __main__ import _gather_features',
        globals={'f': f, 'x': x, 'y': y})

    print(f'transpose_and_gather_features: {t0.timeit(100).mean * 1e6:>6.1f} us')
    print(f'gather_features              : {t1.timeit(100).mean * 1e6:>6.1f} us')

    t1 = benchmark.Timer(
        stmt='_gather_features(f, x, y).sum().backward()',
        setup='from __main__ import _gather_features',
        globals={'f': f.clone().requires_grad_(), 'x': x, 'y': y})

    t0 = benchmark.Timer(
        stmt='transpose_and_gather_features(f, ind).sum().backward()',
        setup='from __main__ import transpose_and_gather_features',
        globals={'f': f.clone().requires_grad_(), 'ind': ind})

    print(f'transpose_and_...(w backward): {t0.timeit(100).mean * 1e6:>6.1f} us')
    print(f'gather_features (w backward) : {t1.timeit(100).mean * 1e6:>6.1f} us')


if __name__ == '__main__':
    test_speed_for_gather_features()
