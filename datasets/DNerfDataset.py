import json
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from datasets.base import NERF_DATASET_STYLE, NERF_DATASETS, NERF_Base_Dataset
from my_ext import ops_3d, utils

NERF_DATASETS['DNeRF'] = {
    'common': {
        'style': 'DNeRFDataset',
        'root': 'DNeRF',
        'use_time': True,
        'downscale': 1,
        'near': 2.,
        'far': 6.0,
        'background': 'white',
    },
    'train': {'split': 'train'},
    'eval': {'split': 'val'},
    'test': {'split': 'test'},
    'DNeRF': ['train', 'eval', 'test'],
}


@NERF_DATASET_STYLE.register()
class DNeRFDataset(NERF_Base_Dataset):

    def __init__(
        self,
        root: Path,
        scene='',
        camera_file='transforms_{}.json',
        split='train',
        img_dir='',
        img_suffix='.png',
        mask_dir='',
        mask_suffix='',
        background='white',
        near=2.,
        far=6.0,
        image_size=None,
        downscale: int = None,
        random_camera=False,
        is_srgb_image=False,
        batch_mode=False,
        camera_radiu_scale=1.0,  # make sure camera are inside the bounding box.
        camera_noise_R=0.,  # Add noise to camera pose
        camera_noise_t=0.,  # Add noise to camera pose
        coord_src='opengl',
        coord_dst='opengl',
        sample_stride=1,
        use_time=False,
        weighted_sample=False,
        with_rays=True,
        num_frames_max=-1,
        **kwargs
    ):
        root = root.joinpath(scene)
        self.root = root
        self.scene = scene
        assert img_suffix in utils.image_extensions
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.ndc = False  # normalized device coordinates
        self.random_camera = random_camera
        self.batch_mode = batch_mode
        self.downscale = downscale
        self.sample_stride = sample_stride
        self.split = split
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)
        self.weighted_sample = weighted_sample

        self.with_rays = with_rays

        fovx = None
        if camera_file.endswith('.json'):
            self.camera_file = camera_file.format(split)
            fovx, self.Tv2w, paths, times = self.load_camera_from_json(root.joinpath(self.camera_file))
        elif camera_file.endswith('.npz'):
            self.camera_file = camera_file
            paths = sorted(list(root.joinpath(img_dir).glob('*' + img_suffix)))
            self.Tv2w, self.Tv2s, self.Ts = self.load_camera_from_npz(
                root.joinpath(camera_file), [int(p.stem) for p in paths])
            self.focal = self.Tv2s[0, 0, 0]
            # self.Ts2v = torch.inverse(self.Tv2s)
            times = None
        else:
            raise NotImplementedError
        if len(paths) > num_frames_max > 0:
            self.Tv2w = self.Tv2w[:num_frames_max]
            paths = paths[:num_frames_max]
            if times is not None:
                times = times[:num_frames_max]
            if self.Tv2s is not None and self.Tv2s.ndim == 3:
                self.Tv2s = self.Tv2s[:num_frames_max]
            if self.Ts is not None and self.Ts.ndim == 3:
                self.Ts = self.Ts[:num_frames_max]
            logging.info(f'[red] only use first {len(paths)} images')

        self.Tv2w = ops_3d.convert_coord_system(self.Tv2w, coord_src, coord_dst, inverse=True)
        # self.Tv2w = ops_3d.convert_coord_system_matrix(self.Tv2w, self.coord_src, self.coord_dst)

        self.images = self.load_images(paths, image_size, downscale, srgb=is_srgb_image)  # shape: [N, H, W, 3]
        if mask_suffix:
            masks = [utils.load_image(root.joinpath(mask_dir, f"{path.stem}{mask_suffix}")) for path in paths]
            masks = np.stack(masks).astype(np.bool_)
            if masks.ndim == 4:
                masks = masks[..., 0]
            self.masks = torch.from_numpy(masks)
            assert self.images.shape[-1] == 3
            self.images = torch.cat([self.images, self.masks[..., None].to(self.images)], dim=-1)
        self.image_size = (self.images.shape[2], self.images.shape[1])
        self.aspect = self.image_size[0] / self.image_size[1]
        self.times = times if use_time else None  # * 2 - 1.
        self.num_frames = len(times)  # single camera
        self.num_cameras = -1
        self.time_ids = torch.arange(self.num_frames)
        self.camera_ids = torch.zeros_like(self.time_ids)

        if fovx is None:
            fovx = ops_3d.focal_to_fov(self.focal, self.images[0])
        self.FoV = torch.tensor([fovx, ops_3d.fovx_to_fovy(fovx, self.aspect)], dtype=torch.float)
        self.background_type = background
        self.init_background(self.images)
        if self.background_type not in ['random', 'random2', 'none']:
            torch.lerp(self.background, self.images[..., :3], self.images[..., -1:], out=self.images[..., :3])

        self.camera_radiu_scale = camera_radiu_scale
        self.Tv2w[:, :3, 3] = (self.Tv2w[:, :3, 3] * camera_radiu_scale + 0)

        self.camera_noise = (float(camera_noise_R), float(camera_noise_t))
        if self.camera_noise != (0., 0.):
            self.Tv2w_origin = self.Tv2w.clone()
            noise_R = torch.randn(len(self.Tv2w), 3) * self.camera_noise[0]
            noise_t = torch.randn(len(self.Tv2w), 3) * self.camera_noise[1]
            self.Tv2w = ops_3d.rigid.lie_to_Rt(noise_R, noise_t) @ self.Tv2w

        self.scene_size = 2.6
        self.scene_center = 0  # [-1.3, 1.3]
        self.complete_transform_matrices(near=near, far=far)
        super().__init__(root, self.images, near=near, far=far, **kwargs)

    def load_camera_from_json(self, camera_file: Path):
        with camera_file.open('r') as f:
            meta = json.load(f)
        cams = []
        paths = []
        times = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            paths.append(self.root.joinpath(self.img_dir, frame['file_path'] + self.img_suffix))
            times.append(frame['time'] if 'time' in frame else float(i) / (len(meta['frames']) - 1))
        fovx = float(meta["camera_angle_x"])
        Tv2w = torch.from_numpy(np.stack(cams, axis=0))
        times = torch.tensor(times, dtype=torch.float)
        return fovx, Tv2w, paths, times

    def load_camera_from_npz(self, camera_file: Path, indices: List[int]):
        meta = np.load(camera_file)
        Tw2s = np.stack([meta[f"world_mat_{i}"].astype(np.float64) for i in indices])
        scales = np.stack([meta[f"scale_mat_{i}"].astype(np.float64) for i in indices])
        Tw2s = Tw2s @ scales
        if False and f"camera_mat_{indices[0]}" in meta:
            Tv2s = np.stack([meta[f"camera_mat_{i}"].astype(np.float64) for i in indices])
            Tv2w = np.linalg.inv(Tw2s) @ Tv2s
        else:
            Tv2w = []
            Tv2s = []
            for i in range(len(Tw2s)):
                out = cv2.decomposeProjectionMatrix(Tw2s[i, :3, :4])
                K = out[0]
                R = out[1]
                t = out[2]

                K = K / K[2, 2]
                intrinsics = np.eye(4)
                intrinsics[:3, :3] = K

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R.transpose()
                pose[:3, 3] = (t[:3] / t[3])[:, 0]
                Tv2s.append(intrinsics)
                Tv2w.append(pose)
            Tv2w = np.stack(Tv2w)
            Tv2s = np.stack(Tv2s)
        return torch.from_numpy(Tv2w), torch.from_numpy(Tv2s)[..., :3, :3], torch.from_numpy(scales)

    def random_ray(self, index, num_rays):
        if self.random_camera:
            cam_ind = torch.randint(0, len(self.images), (num_rays,))
        else:
            cam_ind = torch.randint(0, len(self.images), (1,)) if index is None else torch.tensor([index])

        y_ind = torch.randint(0, self.image_size[1], (num_rays,))
        x_ind = torch.randint(0, self.image_size[0], (num_rays,))
        xy = torch.stack([x_ind, y_ind], dim=-1).float()

        Tv2s = self.Tv2s if self.Tv2s.ndim == 2 else self.Tv2s[cam_ind]
        rays = ops_3d.get_rays(Tv2s, self.Tv2w[cam_ind], xy=xy, normalize=True, offset=0.5, stack_mode=False)
        pixels = self.images[cam_ind, y_ind, x_ind, :]
        inputs = {'rays_o': rays[0], 'rays_d': rays[1], 'background': self.get_background(pixels, x_ind, y_ind)}
        if self.background_type == 'random':
            torch.lerp(inputs['background'], pixels[..., :3], pixels[..., -1:], out=pixels[..., :3])
        targets = {'images': pixels}
        transform_matrices = {
            'Tw2v': self.Tw2v[cam_ind],
            'Tw2c': (self.Tv2c if self.Tv2c.ndim == 2 else self.Tv2c[cam_ind]) @ self.Tw2v[cam_ind],
            'size': self.image_size,
            'index': None if self.random_camera else cam_ind,
            'campos': self.Tv2w[cam_ind, :3, 3],
            'cam_id': 0,
            'FoV': self.FoV,
        }
        if self.times is not None:
            inputs['t'] = self.times[cam_ind]  # .expand_as(pixels[:, 0:1])
            if self.split == 'train':
                inputs['time_id'] = cam_ind

        return inputs, targets, transform_matrices

    def camera_ray(self, index, batch_size=None):
        s = self.sample_stride
        if batch_size is not None:
            index = torch.randint(0, len(self.images), (batch_size,))
        Tv2s = self.Tv2s if self.Tv2s.ndim == 2 else self.Tv2s[index]
        inputs = {}
        if self.with_rays:
            rays = ops_3d.get_rays(Tv2s, self.Tv2w[index], size=self.image_size, normalize=True, offset=0.5,
                sample_stride=s)
            inputs['rays_o'] = rays[0]
            inputs['rays_d'] = rays[1]
        image = self.images[index, ::s, ::s]
        infos = {
            'Tw2v': self.Tw2v[index],
            'Tv2c': (self.Tv2c if self.Tv2c.ndim == 2 else self.Tv2c[index]),
            'Tv2s': Tv2s,
            'size': (image.shape[-2], image.shape[-3]),
            'index': index,
            'campos': self.Tv2w[index, :3, 3],
            'cam_id': self.camera_ids[index],
            'FoV': self.FoV,
        }
        inputs['background'] = self.get_background(image, slice(0, 0, s), slice(0, 0, s))
        if self.background_type in ['random', 'random2'] and image.shape[-1] == 4:
            torch.lerp(inputs['background'], image[..., :3], image[..., -1:], out=image[..., :3])
        targets = {'images': image}
        if self.times is not None:
            inputs['t'] = self.times[index]  # .expand_as(image[..., 0:1])
            if self.split == 'train':
                inputs['time_id'] = self.time_ids[index]
        return inputs, targets, infos

    def __getitem__(self, index=None):
        if isinstance(index, int):
            return self.camera_ray(index)
        elif isinstance(index, tuple):
            if self.batch_mode:
                return self.camera_ray(None, index[1])
            else:
                return self.random_ray(None, index[1])
        else:
            raise RuntimeError(f"{index}")

    def extra_repr(self):
        s = [
            f"img_dir: {self.img_dir}, img_suffix: {self.img_suffix}" if self.img_dir else None,
            f"mask_dir: {self.mask_dir}, mask_suffix: {self.mask_suffix}" if self.mask_suffix else None,
            f"camera_file: {self.camera_file}, coord system: {self.coord_src}→{self.coord_dst}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}, focal={utils.float2str(self.focal)}, split: {self.split}",
            f"background={self.background_type}",
            f"camera_radiu_scale={self.camera_radiu_scale}" if self.camera_radiu_scale != 1.0 else None,
            f"camera noise: {self.camera_noise}" if self.camera_noise != (0., 0.) else None,
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
        ]
        return super().extra_repr() + s
