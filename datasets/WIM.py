import logging
from pathlib import Path
import json

import torch
from torch import Tensor
import numpy as np

from datasets.base import NERF_DATASETS, NERF_Base_Dataset, NERF_DATASET_STYLE
from my_ext import ops_3d, utils

NERF_DATASETS['WIM'] = {
    'common': {
        'style': 'WIM',
        'root': 'WIM',
        'scene': 'atlas',
        'test_cameras': [0, 10],
        'frame_ranges': [0, 300],
        'coord_src': 'opengl',
        'coord_dst': 'opengl',
    },
    'train': {'split': 'train'},
    'val': {'split': 'test', 'num_random_sample': 20},
    'test': {'split': 'test'},
    'WIM': ['train', 'val', 'test']
}


@NERF_DATASET_STYLE.register('WIM')
class WatchItMoveDataset(NERF_Base_Dataset):
    def __init__(
        self,
        root: Path,
        scene='atlas',
        split='train',
        image_size=None,
        downscale=1,
        with_rays=True,
        batch_mode=False,
        coord_src='opengl',
        coord_dst='opengl',
        background='white',
        test_cameras=(0, 10),
        frame_ranges=(0, 300),
        num_random_sample=-1,
        **kwargs
    ):
        root = Path(root).expanduser().joinpath(scene)
        self.scene = scene
        self.with_rays = with_rays
        self.batch_mode = batch_mode
        self.split = split
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)
        camera_indices = [idx for idx in range(20) if (idx not in test_cameras) == (split == 'train')]

        Tv2ws = []
        # Tw2vs = []

        cx_cy_fx_fy = None
        for cam_idx in camera_indices:
            with root.joinpath(f'cam_{cam_idx:03d}.json').open('r') as f:
                camera_info = json.load(f)
                Tv2w = torch.tensor(camera_info['camera_data']['cam2world']).T
                # Tw2v = camera_info['camera_data']['camera_view_matrix']  # .T
                # Tv2w_ = ops_3d.look_at(
                #     torch.tensor(camera_info['camera_data']['camera_look_at']['eye'], dtype=torch.float),
                #     torch.tensor(camera_info['camera_data']['camera_look_at']['at'], dtype=torch.float),
                #     torch.tensor(camera_info['camera_data']['camera_look_at']['up'], dtype=torch.float),
                #     inv=True,
                # )
                # print((Tv2w_ - Tv2w).abs().max())
                Tv2ws.append(Tv2w)
                # Tw2vs.append(Tw2v)
                assert camera_info['camera_data']['width'] == 800 and camera_info['camera_data']['height'] == 800
                self.image_size = (camera_info['camera_data']['width'], camera_info['camera_data']['height'])
                K = camera_info['camera_data']['intrinsics']
                if cx_cy_fx_fy is None:
                    cx_cy_fx_fy = K['cx'], K['cy'], K['fx'], K['fy']
                else:
                    assert cx_cy_fx_fy == (K['cx'], K['cy'], K['fx'], K['fy'])
                # location_world
                # quaternion_world_xyzw
                # scene_center_3d_box
                # scene_max_3d_box
                # scene_min_3d_box
        assert cx_cy_fx_fy is not None and cx_cy_fx_fy[2] == cx_cy_fx_fy[3], cx_cy_fx_fy
        self.aspect = self.image_size[0] / self.image_size[1]
        self.focal = cx_cy_fx_fy[2]
        fovy = ops_3d.focal_to_fov(self.focal, self.image_size[1])
        self.FoV = torch.tensor([ops_3d.fovx_to_fovy(fovy, 1. / self.aspect), fovy], dtype=torch.float)
        self.Tv2w = torch.stack(Tv2ws)
        self.Tv2w = ops_3d.convert_coord_system(self.Tv2w, self.coord_src, self.coord_dst, inverse=True)
        # self.Tv2w = ops_3d.convert_coord_system_matrix(self.Tv2w, self.coord_src, self.coord_dst)
        # self.Tw2v = torch.tensor(Tw2vs).transpose(-1, -2)
        self.complete_transform_matrices()
        self.num_cameras = len(self.Tv2w)

        self.frame_ranges = frame_ranges  # [start, end, step]
        assert len(frame_ranges) in [2, 3]
        image_paths = []
        times = []
        time_ids = []
        camera_ids = []
        for i, fid in enumerate(range(*self.frame_ranges)):
            for k, cid in enumerate(camera_indices):
                image_paths.append(root.joinpath(f"frame_{fid:0>5d}_cam_{cid:0>3d}.png"))
                times.append((fid - self.frame_ranges[0]) / (self.frame_ranges[1] - self.frame_ranges[0]))
                time_ids.append(i)
                camera_ids.append(k)
        self.images = self.load_images(image_paths[:1], image_size, downscale)
        self.image_size = (self.images.shape[-2], self.images.shape[-3])
        self.downscale = downscale
        assert downscale == 1

        self.background_type = background
        self.init_background(self.images)
        if self.background_type in ['black', 'white', 'checker']:
            torch.lerp(self.background, self.images[..., :3], self.images[..., 3:], out=self.images[..., :3])

        self.time_ids = torch.tensor(time_ids, dtype=torch.long)
        self.camera_ids = torch.tensor(camera_ids, dtype=torch.long)
        self.num_frames = len(image_paths) // self.num_cameras
        self.times = torch.tensor(times, dtype=torch.float)  # [0, 1]

        self.num_random_sample = num_random_sample
        if num_random_sample > 0:
            logging.info(f"random choose {num_random_sample} from {len(image_paths)} images")
            index = np.random.choice(len(image_paths), min(num_random_sample, len(image_paths)), replace=False)
            image_paths = [image_paths[i] for i in index]
            index = torch.from_numpy(index)
            self.time_ids = self.time_ids[index]
            self.camera_ids = self.camera_ids[index]
            self.times = self.times[index]

        self.scene_size = 2.6
        self.scene_center = 0
        super().__init__(root, [img_path.name for img_path in image_paths], **kwargs)

    def random_ray(self, index, num_rays):
        # if self.random_camera:
        #     img_idx = torch.randint(0, len(self.images), (num_rays,))
        # else:
        img_idx = torch.randint(0, len(self.samples), (1,)) if index is None else torch.tensor([index])
        cam_idx = self.camera_ids[img_idx]
        y_ind = torch.randint(0, self.image_size[1], (num_rays,))
        x_ind = torch.randint(0, self.image_size[0], (num_rays,))
        xy = torch.stack([x_ind, y_ind], dim=-1).float()

        Tv2s = self.Tv2s if self.Tv2s.ndim == 2 else self.Tv2s[cam_idx]
        rays = ops_3d.get_rays(Tv2s, self.Tv2w[cam_idx], xy=xy, normalize=True, offset=0.5, stack_mode=False)
        pixels = utils.load_image(self.root.joinpath(self.samples[img_idx.item()]))
        pixels = torch.from_numpy(pixels).float() / 255.
        pixels = pixels[y_ind, x_ind]
        # pixels = self.images[img_idx, y_ind, x_ind, :]
        inputs = {'rays_o': rays[0], 'rays_d': rays[1], 'background': self.get_background(pixels, x_ind, y_ind)}
        if self.background_type == 'random':
            torch.lerp(inputs['background'], pixels[..., :3], pixels[..., -1:], out=pixels[..., :3])
        targets = {'images': pixels}
        infos = {
            'Tw2v': self.Tw2v[cam_idx],
            'Tw2c': self.Tv2c,
            'size': self.image_size,
            'index': img_idx,
            'campos': self.Tv2w[cam_idx, :3, 3],
            'FoV': self.FoV,
        }
        if self.times is not None:
            inputs['t'] = self.times[img_idx].expand_as(pixels[:, 0:1])
            inputs['time_id'] = self.time_ids[index]  # .expand_as(pixels[..., 0:1])
            infos['cam_id'] = cam_idx
        return inputs, targets, infos

    def camera_ray(self, index, batch_size=None):
        s = 1
        if batch_size is not None:
            index = torch.randint(0, len(self), (batch_size,))
        if isinstance(index, Tensor):
            index = index.item()
        cam_idx = self.camera_ids[index]
        assert (0 <= index < len(self.times)) and (0 <= self.time_ids[index] < self.num_frames) and (
            cam_idx < len(self.Tv2w))
        Tv2s = self.Tv2s
        inputs = {}
        if self.with_rays:
            rays = ops_3d.get_rays(Tv2s, self.Tv2w[cam_idx], size=self.image_size, normalize=True, offset=0.5,
                                   sample_stride=s)
            inputs['rays_o'] = rays[0]
            inputs['rays_d'] = rays[1]
        image = utils.load_image(self.root.joinpath(self.samples[index]), size=self.image_size)
        image = torch.from_numpy(image).float() / 255.

        infos = {
            'Tw2v': self.Tw2v[cam_idx],
            'Tv2c': self.Tv2c,
            'Tv2s': Tv2s,
            'size': (image.shape[-2], image.shape[-3]),
            'index': index,
            'campos': self.Tv2w[cam_idx, :3, 3],
            'FoV': self.FoV,
        }
        inputs['background'] = self.get_background(image, slice(0, 0, s), slice(0, 0, s))
        # if self.background_type == 'random' and image.shape[-1] == 4:
        torch.lerp(inputs['background'], image[..., :3], image[..., -1:], out=image[..., :3])
        targets = {'images': image}
        if self.times is not None:
            inputs['t'] = self.times[index]  # .expand_as(image[..., 0:1])
            inputs['time_id'] = self.time_ids[index]  # .expand_as(image[..., 0:1])
            infos['cam_id'] = cam_idx
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
            raise RuntimeError()

    def extra_repr(self):
        s = [
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}",
            f"background={self.background_type}",
            f"frames: [{':'.join(str(i) for i in self.frame_ranges)}], "
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
        ]
        return super().extra_repr() + s

    def get_image(self, index: int):
        image = utils.load_image(self.root.joinpath(self.samples[index]), size=self.image_size)
        image = torch.from_numpy(image).float() / 255.
        background = self.get_background(image)
        torch.lerp(background, image[..., :3], image[..., -1:], out=image[..., :3])
        return image
