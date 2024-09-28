import pickle
from pathlib import Path

import cv2
import torch
from torch import Tensor
import numpy as np

from datasets.base import NERF_DATASETS, NERF_Base_Dataset, NERF_DATASET_STYLE
from my_ext import ops_3d, utils

NERF_DATASETS['ZJU_MoCap'] = {
    'common': {
        'style': 'ZJU_MoCap',
        'root': 'ZJU-MoCap',
        'scene': '313',
        'coord_src': 'colmap',
    },
    'train': {'split': 'train'},
    'val': {'split': 'test', 'num_val': 20},
    'test': {'split': 'test'},
    'ZJU_MoCap': ['train', 'val', 'test']
}
NERF_DATASETS['ZJU_MoCap_2'] = {
    'common': {
        'style': 'ZJU_MoCap_2',
        'root': 'ZJU-MoCap/zju/cache512',
        'scene': '377',
        'batch_mode': True,
    },
    'train': {'pickle_path': 'cache_train.pickle'},
    'test': {'pickle_path': 'cache_test.pickle'},
    'ZJU_MoCap_2': ['train', 'test']
}


@NERF_DATASET_STYLE.register('ZJU_MoCap')
class ZJUMoCapDataset(NERF_Base_Dataset):
    def __init__(
        self,
        root: Path,
        scene='313',
        num_max_frames=300,
        num_max_cameras=-1,
        split='train',
        image_size=None,
        downscale=1,
        with_rays=True,
        batch_mode=False,
        coord_src='opengl',
        coord_dst='opengl',
        use_perspective_v2=False,
        background='white',
        mask_dir='mask',
        train_camera_ids=(0, 6, 12, 19),  # start from 0
        num_val=-1,
        **kwargs
    ):
        self.with_rays = with_rays
        self.batch_mode = batch_mode
        self.split = split
        self.mask_dir = mask_dir
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)

        root = Path(root).expanduser().joinpath(f"CoreView_{scene}")
        annots = np.load(root.joinpath('annots.npy'), allow_pickle=True).item()

        camera_infos = annots['cams']
        K = torch.tensor(np.array(camera_infos['K']))
        R = torch.tensor(np.array(camera_infos['R']))
        T = torch.tensor(np.array(camera_infos['T']))
        D = torch.tensor(np.array(camera_infos['D']))
        self.num_cameras = len(K)
        coord_scale = 0.001
        self.Tw2v = torch.zeros(self.num_cameras, 4, 4)
        self.Tw2v[:, :3, :3] = R
        self.Tw2v[:, :3, 3:] = T * coord_scale
        self.Tw2v[:, 3, 3] = 1
        self.Tv2s = K

        image_infos = annots['ims']
        image_paths = []
        self.num_frames = len(image_infos) if num_max_frames < 0 else min(len(image_infos), num_max_frames)
        time_ids = []
        camera_ids = []
        for fid in range(self.num_frames):
            for cid, image_path in enumerate(image_infos[fid]['ims']):
                if (split == 'train') == (cid in train_camera_ids):
                    image_paths.append(image_path)
                    time_ids.append(fid)
                    camera_ids.append(cid)
        self.time_ids = torch.tensor(time_ids)
        self.camera_ids = torch.tensor(camera_ids)
        self.times: Tensor = self.time_ids.float() / self.num_frames  # [0, 1]
        if split != 'train' and num_val > 0:
            index = np.random.choice(len(image_paths), num_val, replace=False)
            image_paths = [image_paths[i] for i in index]
            index = torch.from_numpy(index)
            self.time_ids = self.time_ids[index]
            self.camera_ids = self.camera_ids[index]
            self.times = self.times[index]

        self.images = self.load_images([root.joinpath(image_paths[0])], image_size, downscale)
        self.background_type = background
        self.init_background(self.images)
        self.image_size = (self.images.shape[-2], self.images.shape[-3])
        self.downscale = downscale
        assert downscale == 1

        self.train_camera_ids = train_camera_ids
        self.aspect = self.image_size[0] / self.image_size[1]
        self.focal = K[:, 0, 0].mean().item()
        self.fovy = ops_3d.focal_to_fov(self.focal, self.image_size[1])
        self.Tw2v = ops_3d.convert_coord_system(self.Tw2v, coord_src, coord_dst, inverse=False)
        self.Tv2c = ops_3d.perspective(self.fovy, self.aspect, n=0.01)
        # self.Tw2v = torch.tensor(Tw2vs).transpose(-1, -2)
        self.complete_transform_matrices()

        self.scene_size = 2.6
        self.scene_center = 0
        super().__init__(root, image_paths, **kwargs)

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
        print(utils.show_shape(pixels))
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
        Tv2s = self.Tv2s[cam_idx]
        inputs = {}
        if self.with_rays:
            rays = ops_3d.get_rays(
                Tv2s, self.Tv2w[cam_idx], size=self.image_size, normalize=True, offset=0.5, sample_stride=s)
            inputs['rays_o'] = rays[0]
            inputs['rays_d'] = rays[1]
        # if self.split == 'train':
        image = utils.load_image(self.root.joinpath(self.samples[index]))
        image = torch.from_numpy(image).float() / 255.

        # else:
        #     image = self.images[index, ::s, ::s]  # [1, H, W, C]
        infos = {
            'Tw2v': self.Tw2v[cam_idx],
            'Tw2c': self.Tv2c @ self.Tw2v[cam_idx],
            'Tv2s': Tv2s,
            'size': (image.shape[-2], image.shape[-3]),
            'index': index,
            'campos': self.Tv2w[cam_idx, :3, 3],
            'FoV': self.FoV,
        }
        inputs['background'] = self.get_background(image, slice(0, 0, s), slice(0, 0, s))
        if self.mask_dir:
            mask = utils.load_image(self.root.joinpath(self.mask_dir, self.samples[index]).with_suffix('.png'))
            mask = torch.from_numpy(mask)
            image = torch.cat([image, mask[..., None]], dim=-1)
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
            # f"img_dir: {self.img_dir}, img_suffix: {self.img_suffix}" if self.img_dir else None,
            # f"meta_file: {self.meta_file}, split: {self.split_file}, scene_file: {self.scene_file}",
            f"mask_dir: {self.mask_dir}" if self.mask_dir else None,
            # f"points file: {self.points_file}",
            # f"camera: {self.camera_dir}/*{self.camera_suffix}, coord system: {self.coord_src}→{self.coord_dst}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}, focal={utils.float2str(self.focal)}",
            f"background={self.background_type}",
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
            f"train_camera_ids: {self.train_camera_ids}, split={self.split}"
            # f"camera_radiu_scale={self.camera_radiu_scale}" if self.camera_radiu_scale != 1.0 else None,
            # f"camera noise: {self.camera_noise}" if self.camera_noise > 0 else None,
        ]
        return super().extra_repr() + s

    def get_image(self, index: int):
        image = utils.load_image(self.root.joinpath(self.samples[index]))
        image = torch.from_numpy(image).float() / 255.
        # background = self.get_background(image)
        # torch.lerp(background, image[..., :3], image[..., -1:], out=image[..., :3])
        return image


@NERF_DATASET_STYLE.register('ZJU_MoCAP_2')
class ZJU_MoCAP_Dataset_pickled(NERF_Base_Dataset):
    def __init__(
        self,
        root,
        scene='377',
        pickle_path='cache_train.pickle',
        frame_ranges=(-1, -1),
        step=1,
        image_size=512,
        compression=True,
        background='none',
        use_perspective_v2=False,
        near=0.1,
        far=1000,
        with_rays=True,
        batch_mode=True,
        coord_src='colmap',
        coord_dst='colmap',
        move_center=True,
        **kwargs
    ):
        self.with_rays = with_rays
        self.batch_mode = batch_mode
        self.move_center = move_center
        self.coord_src = ops_3d.coordinate_system[coord_src.lower()]
        self.coord_dst = ops_3d.coordinate_system[coord_dst.lower()]
        ops_3d.set_coord_system(self.coord_dst)

        self.scene = scene
        self.pickle_path = pickle_path
        with open(root.joinpath(scene, pickle_path), 'rb') as file:
            data = pickle.load(file)

        images = []
        intrinsics = []
        poses = []
        times = []
        camera_ids = []
        time_ids = []
        frame_indies = np.unique(data['frame_id'])
        imgs_per_cam = len(frame_indies)
        fid_max = frame_indies.max()
        id_min = frame_indies.min() if frame_ranges[0] < 0 else max(frame_ranges[0], frame_indies.min())
        id_max = frame_indies.max() + 1 if frame_ranges[1] < 0 else min(frame_ranges[1], frame_indies.max() + 1)
        self.frame_ranges = (id_min, id_max)  # [min, max)
        self.num_frames = len([fid for fid in frame_indies if id_min <= fid < id_max])
        self.frame_id_max = fid_max

        camera_indies = np.unique(data['camera_id'])
        self.num_cameras = len(camera_indies)
        counter = 0
        img_scale = 1
        for f_id in range(0, imgs_per_cam, step):
            if not (id_min <= frame_indies[f_id] < id_max):
                continue
            ids = []
            for k, c_id in enumerate(camera_indies):  # type: int, int
                index = c_id * imgs_per_cam + f_id
                times.append((data['frame_id'][f_id] - id_min) / fid_max)
                image = data['img'][index]
                mask = data['mask'][index]
                if compression:
                    import blosc
                    image = blosc.unpack_array(image)
                    mask = blosc.unpack_array(mask)[None, :, :]

                image = np.concatenate((image, mask.astype(np.uint8) * 255), axis=0)
                image = np.transpose(image, [1, 2, 0])

                if image.shape[0] != image_size:
                    img_scale = image_size / image.shape[0]
                    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

                intrinsic = data["camera_intrinsic"][index] * img_scale
                intrinsic[2, 2] = 1.
                if self.move_center:
                    H, W, _ = image.shape
                    M = np.array([[1, 0, W * 0.5 - intrinsic[0, 2]], [0, 1, H * 0.5 - intrinsic[1, 2]]])
                    image = cv2.warpAffine(image, M, (W, H))
                    intrinsic[0, 2] = 0.5 * W
                    intrinsic[1, 2] = 0.5 * H

                image = torch.from_numpy(image)
                images.append(image)
                camera_ids.append(k)
                time_ids.append(f_id)

                if len(poses) < len(camera_indies):
                    intrinsics.append(intrinsic)

                    coordinate_scale = 1.5  # NOTE: From WIM

                    rot = data["camera_rotation"][index]
                    trans = data["camera_translation"][index] / coordinate_scale
                    pose = np.concatenate([np.concatenate([rot, trans], axis=-1), np.array([[0, 0, 0, 1]])], axis=0)
                    pose = np.linalg.inv(pose)
                    poses.append(pose)

                ids.append(counter)
                counter += 1

        self.images = torch.stack(images, dim=0)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.time_ids = torch.tensor(time_ids, dtype=torch.long)
        self.time_ids = torch.unique(self.time_ids, return_inverse=True)[1]
        self.camera_ids = torch.tensor(camera_ids, dtype=torch.long)

        self.image_size = (self.images.shape[2], self.images.shape[1])
        self.aspect = self.image_size[0] / self.image_size[1]
        self.downscale = img_scale

        self.Tv2w = torch.from_numpy(np.array(poses)).to(torch.float32)
        self.Tv2s = torch.from_numpy(np.array(intrinsics))

        fx, fy, cx, cy = self.Tv2s[:, 0, 0], self.Tv2s[:, 1, 1], self.Tv2s[:, 0, 2], self.Tv2s[:, 1, 2]
        fovx = ops_3d.focal_to_fov(fx, self.image_size[0])
        fovy = ops_3d.focal_to_fov(fy, self.image_size[1])
        self.FoV = torch.stack([fovx, fovy], dim=-1).float()

        self.Tv2w = ops_3d.convert_coord_system(self.Tv2w, self.coord_src, self.coord_dst, inverse=True)

        self.Tv2c = ops_3d.perspective(self.image_size, fx, fy, cx, cy, n=near, f=far)
        self.scene_size = 2.6
        self.scene_center = 0  # [-1.3, 1.3]
        self.complete_transform_matrices(near=near, far=far)

        self.background_type = background
        self.init_background(self.images)
        if self.background_type not in ['random', 'random2', 'none']:
            self.images[..., :3] = torch.where(self.images[..., -1:] >= 128, self.images[..., :3], self.background)
        super().__init__(root, self.images, near=near, far=far, **kwargs)

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
        pixels = self.images[index].float() / 255.
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
            'FoV': self.FoV[cam_idx],
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
        Tv2s = self.Tv2s if self.Tv2s.ndim == 2 else self.Tv2s[cam_idx]
        Tv2w = self.Tv2w[cam_idx]
        inputs = {}
        if self.with_rays:
            print(utils.show_shape(Tv2s, self.Tv2w[cam_idx]))
            rays = ops_3d.get_rays(Tv2s, Tv2w, size=self.image_size, normalize=True, offset=0.5, sample_stride=s)
            inputs['rays_o'] = rays[0]
            inputs['rays_d'] = rays[1]
        image = self.images[index, ::s, ::s].float() / 255.  # [1, H, W, C]
        infos = {
            'Tw2v': self.Tw2v[cam_idx],
            'Tw2c': self.Tv2c[cam_idx] @ self.Tw2v[cam_idx],
            'Tv2s': Tv2s,
            'size': (image.shape[-2], image.shape[-3]),
            'index': index,
            'campos': self.Tv2w[cam_idx, :3, 3],
            'FoV': self.FoV[cam_idx],
        }
        inputs['background'] = self.get_background(image, slice(0, 0, s), slice(0, 0, s))
        if self.background_type in ['random', 'random2']:
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
            f"scene={self.scene}, pickle_path={self.pickle_path}"
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}",
            # f"focal={utils.float2str(self.focal)}"
            f"background={self.background_type}",
            f"frame_ranges: [{self.frame_ranges[0]}, {self.frame_ranges[1]}), "
            f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
            f"frame_index_max={self.frame_id_max}",
            # f"train_camera_ids: {self.train_camera_ids}, split={self.split}"
            f"move_view_center=True" if self.move_center else None
        ]
        return super().extra_repr() + s
