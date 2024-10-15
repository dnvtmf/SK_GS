from pathlib import Path
from typing import Tuple, Union, Optional

import cv2
import numpy as np
import torch
from torch import Tensor

from my_ext import ops_3d, utils
from .dataset_base import Dataset
from my_ext.utils import Registry, srgb_to_rgb, load_image

NERF_DATASETS = Registry(ignore_case=True)
NERF_DATASET_STYLE = Registry(ignore_case=True)


class NERF_Base_Dataset(Dataset):
    Tv2w: Tensor = None
    """ view   space to world  space"""
    Tw2v: Tensor = None  # world  space to view   space
    Tv2s: Tensor = None  # view   space to screen space
    Ts2v: Tensor = None  # screen space to view   space
    Tv2c: Tensor = None  # view   space to clip   space
    Ts: Tensor = None  # scale matrix
    FoV: Tensor
    """field of view: radians, e.g. 0.25*pi, shape: [2] or [N, 2]"""
    image_size: Tuple[int, int]  # WxH
    focal: Union[float, Tensor] = None  # focal length
    aspect: float
    images: Tensor
    background_type = 'none'
    background: Optional[Tensor] = None
    coord_src: str = 'opengl'
    """The source coordinates system, defalut: opengl"""
    coord_dst: str = 'opengl'
    """The destination coordinate system, defalut: opengl"""

    def __init__(self, root: Path, samples=None, near=0.1, far=1000., **kwargs):
        self.near_far = (near, far)
        super().__init__(root, samples, **kwargs)

    def complete_transform_matrices(self, near=0.01, far=1000.):
        assert self.image_size is not None
        self.aspect = self.image_size[0] / self.image_size[1]
        if self.FoV is None:
            assert self.focal is not None, f"Can not calculate fovy"
            fovx = ops_3d.focal_to_fov(self.focal, self.image_size[0])
            fovy = ops_3d.focal_to_fov(self.focal, self.image_size[1])
            if isinstance(fovx, Tensor):
                self.FoV = torch.stack([fovx, fovy], dim=-1)
            else:
                self.FoV = torch.tensor([fovx, fovy], dtype=torch.float)
        if self.focal is None:
            self.focal = ops_3d.fov_to_focal(self.FoV[..., 1], self.image_size[1])

        if self.Tv2w is None:
            assert self.Tw2v is not None
            self.Tv2w = torch.inverse(self.Tw2v)

        if self.Tw2v is None:
            assert self.Tv2w is not None
            self.Tw2v = torch.inverse(self.Tv2w)
        self.Tv2w = self.Tv2w.float()
        self.Tw2v = self.Tw2v.float()

        if self.Tv2s is None and self.Ts2v is None:
            self.Tv2s = ops_3d.camera_intrinsics(self.focal, size=self.image_size)
            self.Ts2v = ops_3d.camera_intrinsics(self.focal, size=self.image_size, inv=True)
        elif self.Tv2s is None:
            self.Tv2s = torch.inverse(self.Ts2v)
        elif self.Ts2v is None:
            self.Ts2v = torch.inverse(self.Tv2s)  # noqa
        self.Tv2s = self.Tv2s.float()
        self.Ts2v = self.Ts2v.float()

        if self.Tv2c is None:
            self.Tv2c = ops_3d.perspective(self.FoV[..., 1], n=near, f=far, size=self.image_size)
        self.Tv2c = self.Tv2c.float()

        if self.Ts is None:
            self.Ts = torch.eye(4)
        self.Ts = self.Ts.float()

    @property
    def near(self):
        return self.near_far[0]

    @property
    def far(self):
        return self.near_far[1]

    def load_images(self, paths: list, image_size=None, downscale: int = None, srgb=False, fp32: bool = True):
        images = []
        for img_path in paths:  # do not sort paths here!!
            img = load_image(img_path)
            if image_size is None and downscale is not None:
                image_size = (img.shape[1] // downscale, img.shape[0] // downscale)
            if image_size is not None and img.size != image_size:
                img = cv2.resize(img, list(map(int, image_size)), interpolation=cv2.INTER_AREA)  # down scale
            if fp32:
                if img.dtype != np.float32:  # LDR image
                    img = torch.from_numpy(img.astype(np.float32) / 255.)
                    if srgb:
                        img = srgb_to_rgb(img)
                else:  # HDR image
                    img = torch.from_numpy(img.astype(np.float32))
            else:
                if img.dtype != np.uint8:
                    img = np.clip(img * 255., 0, 255).astype(np.uint8)
                img = torch.from_numpy(img)
            images.append(img)
        return torch.stack(images, dim=0)

    def extra_repr(self):
        # theta_range = np.round(np.rad2deg(self.theta_range), 3)
        # phi_range = np.round(np.rad2deg(self.phi_range), 3)
        return [
            f"near={self.near}, far={self.far}",
            # f"camera range: raduis={self.radius_range}, theta={theta_range}, phi={phi_range}",
        ]

    def get_background(self, pixels: Tensor, x_ind=None, y_ind=None) -> Optional[Tensor]:
        if self.background_type == 'none':
            return None
        elif self.background_type == 'black':
            return pixels.new_zeros(1)
        elif self.background_type == 'white':
            return pixels.new_tensor(255 if pixels.dtype == torch.uint8 else 1.)
        elif self.background_type == 'reference':
            return pixels[..., :3].clone()
        elif self.background_type == 'random':
            return torch.rand_like(pixels[..., :3])
        elif self.background_type == 'random2':
            if pixels.dtype == torch.uint8:
                return torch.randint(0, 255, (3,), dtype=pixels.dtype, device=pixels.device)
            else:
                return torch.rand((3,), dtype=pixels.dtype, device=pixels.device)
        elif self.background_type == 'checker':
            return self.background[y_ind, x_ind, :3].expand_as(pixels[..., :3])
        else:
            raise ValueError()

    def init_background(self, images: Tensor):
        if self.background_type == 'white':
            self.background = images.new_tensor(255 if images.dtype == torch.uint8 else 1)
        elif self.background_type == 'black':
            self.background = images.new_tensor(0)
        elif self.background_type == 'reference':
            self.background = images[..., :3]
        elif self.background_type == 'random':
            if images.dtype == torch.uint8:
                self.background = torch.randint_like(images[0, ..., :3], 0, 255)
            else:
                self.background = torch.rand_like(images[0, ..., :3])
        elif self.background_type == 'random2':
            if images.dtype == torch.uint8:
                self.background = torch.randint(0, 255, (1, 1, 3), dtype=images.dtype)  # .expand_as(images)
            else:
                self.background = torch.rand(1, 1, 3, dtype=images.dtype)  # .expand_as(images)

        elif self.background_type == 'checker':
            N, H, W, C = images.shape
            self.background = torch.from_numpy(utils.image_checkerboard((H, W), 8)).to(images)
        elif self.background_type == 'none':
            self.background = None
        else:
            raise NotImplementedError(f"background type \"{self.background_type}\" is not support")

    def get_image(self, index: int):
        image = self.images[index]
        if self.background_type in ['random', 'random2'] and image.shape[-1] == 4:
            background = self.get_background(image)
            image = image.clone()
            torch.lerp(background, image[..., :3], image[..., -1:], out=image[..., :3])
        return image

    def get_fovy(self, index: int):
        return self.FoV[1] if self.FoV.ndim == 1 else self.FoV[index, 1]


def find_best_img_dir(root: Path, img_dir: str, goal: float):
    """选择最接近目标下采样倍数的图片目录"""
    downscales = []
    if root.joinpath(img_dir).exists():
        downscales.append((1., img_dir))
    for img_dir_s in root.glob(f"{img_dir}_x*"):
        if img_dir_s.is_dir():
            try:
                scale = float(img_dir_s.name[len(f"{img_dir}_x"):])
                downscales.append((scale, img_dir_s.name))
            except ValueError:
                pass
    assert len(downscales) > 0
    downscales = sorted(downscales)
    best = None
    best_img_dir = img_dir
    for scale, img_dir in downscales:
        if scale == goal:
            best, best_img_dir = scale, img_dir
            break
        if scale > goal:
            if best is None:
                best, best_img_dir = scale, img_dir
            break
        best, best_img_dir = scale, img_dir
    return goal / best, best_img_dir


class DynamceSceneDataset(NERF_Base_Dataset):
    times: Tensor
    camera_ids: Tensor
    time_ids: Tensor
    num_cameras: int
    """ < 0 表示每张图片都有一个相机位姿, > 0 表示相机的数量"""
    num_frames: int
    scene: str

    def get_fovy(self, index: int):
        if self.num_cameras > 0:
            return self.FoV[self.camera_ids[index], 1]
        else:
            return self.FoV[1] if self.FoV.ndim == 1 else self.FoV[index, 1]


if __name__ == '__main__':
    db = NERF_Base_Dataset(Path('.'))
    print(db)
