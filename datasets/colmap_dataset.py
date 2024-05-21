import math
import os
import random
from pathlib import Path
from typing import NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn
import json

from datasets.base import NERF_DATASET_STYLE, NERF_Base_Dataset, NERF_DATASETS
from my_ext import utils, ops_3d
from my_ext.utils.io.colmap import (
    read_points3D_text, read_points3D_binary, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, CameraInfo, readColmapCameras,
)

NERF_DATASETS['Mip360'] = {
    'common': {
        'style': 'ColmapDataset',
        'colmap_dir': "sparse/0",
        'img_dir': "images",
        'root': 'NeRF/Mip360',
        'coord_src': 'colmap',
        'coord_dst': 'colmap',
        'background': 'black',
        'scene': 'bicycle'
    },
    'all': {'split': 'train', 'split_eval': False},
    'train': {'split': 'train', 'split_eval': True},
    'test': {'split': 'test', 'split_eval': True},
    'Mip360': ['train', 'test', 'test']
}


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


class Camera_2(nn.Module):
    def __init__(
        self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
        image_name, uid,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
        data_device="cpu"
    ):
        super(Camera_2, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[1]
        self.image_height = self.original_image.shape[0]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((self.image_height, self.image_width, 1), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image  # .permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1)  # .permute(2, 0, 1)


WARNED = False


def loadCam(downscale, uid, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if downscale in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * downscale)), round(orig_h / (resolution_scale * downscale))
    else:  # should be a type that converts to float
        if downscale == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / downscale

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[..., :3]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[..., 3:4]

    return Camera_2(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
        FoVx=cam_info.FovX, FoVy=cam_info.FovY,
        image=gt_image, gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name, uid=uid, data_device='cpu')


def cameraList_from_camInfos(cam_infos, resolution_scale, downscale):
    camera_list = []

    for idx, c in enumerate(cam_infos):
        camera_list.append(loadCam(downscale, idx, c, resolution_scale))

    return camera_list


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


@NERF_DATASET_STYLE.register()
class ColmapDataset(NERF_Base_Dataset):
    def __init__(
        self,
        root: Path,
        scene='',
        img_dir='images',
        colmap_dir='sparse/0',
        shuffle=False,
        downscale=-1,
        split='train',
        coord_src='colmap',
        coord_dst='opengl',
        background='black',
        is_hypernerf_dataset=False,
        split_file='',
        meta_file='',
        split_eval=False,
        split_parts=8,
        near=0.01,
        far=100.,
        **kwargs
    ):
        self.scene = scene
        root = root.joinpath(scene)
        self.train_cameras = {}
        self.test_cameras = {}
        self.coord_src = coord_src
        self.coord_dst = coord_dst
        ops_3d.set_coorder_system(self.coord_dst)
        scene_info = self.read_colmap_camears(root.joinpath(colmap_dir), root.joinpath(img_dir),
            split_eval=split_eval and not is_hypernerf_dataset, split_part=split_parts)
        # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
        #     'wb') as dest_file:
        #     dest_file.write(src_file.read())

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.split_file = split_file
        self.split = split
        self.downscale = downscale
        camera_ids = []
        time_ids = []
        max_time_id = 0
        if is_hypernerf_dataset:
            with root.joinpath(split_file).open('r') as f:
                dataset = json.load(f)
                if len(dataset['val_ids']) > 0:
                    image_names = dataset['train_ids' if self.split == 'train' else 'val_ids']
                else:
                    if self.split == 'train':
                        image_names = [img_name for i, img_name in enumerate(dataset['ids']) if i % 4 != 3]
                    else:
                        image_names = [img_name for i, img_name in enumerate(dataset['ids']) if i % 4 == 3]
            assert len(image_names) > 0, f"Can not found any images for split {split}"

            train_cam_infos = []
            test_cam_infos = []
            for camera_info in scene_info.train_cameras:
                if (camera_info.image_name in image_names) == (self.split == 'train'):
                    train_cam_infos.append(camera_info)
                else:
                    test_cam_infos.append(camera_info)
            scene_info = SceneInfo(
                point_cloud=scene_info.point_cloud,
                train_cameras=train_cam_infos,
                test_cameras=test_cam_infos,
                nerf_normalization=scene_info.nerf_normalization,
                ply_path=scene_info.ply_path
            )

            self.meta_file = meta_file
            with root.joinpath(meta_file).open('r') as f:
                metadata = json.load(f)
                for k, v in metadata.items():
                    max_time_id = max(max_time_id, v['time_id'])
            cameras_infos = scene_info.train_cameras
            for i, cameras_info in enumerate(cameras_infos):
                img_name = cameras_info.image_name
                camera_ids.append(metadata[img_name]['camera_id'])
                time_ids.append(metadata[img_name]['time_id'])
        else:
            cameras = scene_info.train_cameras if self.split == 'train' else scene_info.test_cameras
            for i, cam_info in enumerate(cameras):
                # assert int(cam_info.image_name) == i, f"i={i}, image name: {cam_info.image_name}"
                time_ids.append(i)
                camera_ids.append(0)
                max_time_id = i

        self.time_ids = torch.tensor(time_ids)
        self.camera_ids = torch.tensor(camera_ids)
        self.num_frames = max_time_id + 1
        self.num_cameras = -(self.camera_ids.max() + 1)  # <0
        self.times = torch.tensor(time_ids) / max_time_id  # * 2 - 1.

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        if self.split == 'train':
            print("Loading Training Cameras")
            cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, downscale)
        else:
            print("Loading Test Cameras")
            cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, downscale)
        self.FoV = torch.tensor([[cam.FoVx, cam.FoVy] for cam in cameras], dtype=torch.float)
        if torch.all((self.FoV - self.FoV[:1]).abs().lt(1e-5)):
            self.FoV = self.FoV[0]
        self.image_size = (cameras[0].image_width, cameras[0].image_height)
        self.aspect = self.image_size[0] / self.image_size[1]
        self.point_cloud = scene_info.point_cloud
        self.Tw2v = torch.stack([camera.world_view_transform.transpose(-1, -2) for camera in cameras])
        self.Tw2v = ops_3d.convert_coord_system(self.Tw2v, self.coord_src, self.coord_dst)
        self.Tv2c = torch.stack([camera.projection_matrix.transpose(-1, -2) for camera in cameras])
        # Tv2c = ops_3d.perspective(fovy=self.FoV[..., 1], size=self.image_size, n=near, f=far)
        # print(self.Tv2c[0] - Tv2c)
        self.complete_transform_matrices(near=near, far=far)
        self.images = torch.stack([camera.original_image for camera in cameras], dim=0)
        self.background_type = background
        self.init_background(self.images)
        super().__init__(root, cameras, near=near, far=far, **kwargs)

    @staticmethod
    def read_colmap_camears(colmap_dir: Path, images_dir: Path, split_eval=False, split_part=8):
        try:
            cameras_extrinsic_file = colmap_dir.joinpath("images.bin")
            cameras_intrinsic_file = colmap_dir.joinpath("cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except FileNotFoundError:
            cameras_extrinsic_file = colmap_dir.joinpath("images.txt")
            cameras_intrinsic_file = colmap_dir.joinpath("cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_dir)
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

        if split_eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % split_part != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % split_part == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

        nerf_normalization = getNerfppNorm(train_cam_infos)

        ply_path = colmap_dir.joinpath("points3D.ply")
        bin_path = colmap_dir.joinpath("points3D.bin")
        txt_path = colmap_dir.joinpath("points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except FileNotFoundError:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)

        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path.as_posix()
        )
        return scene_info

    def __getitem__(self, index=None):
        if isinstance(index, tuple):
            index = random.randrange(len(self.samples))
        if index is None:
            index = random.randrange(len(self.samples))
        camera: Camera_2 = self.samples[index]
        inputs = {'background': self.get_background(camera.original_image)}
        targets = {'images': camera.original_image}
        infos = {
            'size': (camera.image_width, camera.image_height),
            'Tw2v': camera.world_view_transform.transpose(-1, -2),
            'Tw2c': camera.full_proj_transform.transpose(-1, -2),
            'campos': camera.camera_center,
            'index': index,
            'FoV': self.FoV[index] if self.FoV.ndim == 2 else self.FoV,
        }
        if self.times is not None:
            inputs['t'] = self.times[index]
            if self.split == 'train':
                inputs['time_id'] = self.time_ids[index]
            infos['cam_id'] = self.camera_ids[index]
        return inputs, targets, infos

    def extra_repr(self):
        focal = self.focal if isinstance(float, (int, float)) else self.focal.mean().item()
        s = [
            # f"img_dir: {self.img_dir}, img_suffix: {self.img_suffix}" if self.img_dir else None,
            # f"meta_file: {self.meta_file}, split: {self.split_file}, scene_file: {self.scene_file}",
            # f"mask_dir: {self.mask_dir}, mask_suffix: {self.mask_suffix}" if self.mask_suffix else None,
            # f"points file: {self.points_file}",
            f"coord system: {self.coord_src}→{self.coord_dst}",
            f"image size{'' if self.downscale is None else f'↓{self.downscale}'}="
            f"{self.image_size[0]} x {self.image_size[1]}, focal={utils.float2str(focal)}",
            f"background={self.background_type}",
            # f"num_frames: {self.num_frames}, num_cameras: {self.num_cameras}",
            # f"camera_radiu_scale={self.camera_radiu_scale}" if self.camera_radiu_scale != 1.0 else None,
            # f"camera noise: {self.camera_noise}" if self.camera_noise > 0 else None,
        ]
        return super().extra_repr() + s


def test_mip360():
    import matplotlib.pyplot as plt
    utils.set_printoptions()
    cfg = {**NERF_DATASETS['Mip360']['common'], **NERF_DATASETS['Mip360']['test']}
    scene = 'bicycle'
    cfg['root'] = Path('~/data', cfg['root']).expanduser()
    cfg['background'] = 'black'
    cfg['scene'] = scene
    cfg['near'] = 0.1
    cfg['far'] = 100.
    # cfg['coord_dst'] = 'opengl'
    db = ColmapDataset(**cfg)
    print(db)
    print(utils.show_shape(db.images, db.Tw2v, db.Tv2c, db.FoV))
    print(utils.show_shape(db[0]))
    # print(utils.show_shape(db.random_ray(0, 1024)))
    # print(utils.show_shape(db.random_ray(None, 1024)))

    # inputs, targets, infos = db.random_ray(0, 1024)
    # aabb = torch.tensor([-1, -1., -1., 1., 1., 1.]).cuda()
    # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
    # from NeRF.networks.ray_sampler import near_far_from_aabb
    # near, far = near_far_from_aabb(rays_o.cuda(), rays_d.cuda(), aabb)
    # print(*near.aminmax(), *far.aminmax())
    # print()

    inputs, targets, infos = db[0]
    # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']

    # plt.subplot(131)
    plt.imshow(targets['images'][..., :3])
    # plt.subplot(132)
    # plt.imshow(inputs['background'].expand_as(targets['images'][..., :3]))
    # plt.subplot(133)
    # plt.imshow(torch.lerp(inputs['background'], targets['images'][..., :3], targets['images'][..., 3:]))
    plt.show()
    fovy = db.FoV[1] if db.FoV.ndim == 1 else db.FoV[0, 1]
    with utils.vis3d:
        utils.vis3d.add_camera_poses(db.Tv2w, None, np.rad2deg(fovy.item()), db.aspect, color=(1, 0, 0), size=0.5)
        # utils.vis3d.add_camera_poses(db.Tv2w_origin, None, np.rad2deg(db.FoV[1].item()), db.aspect, 0.5, (0, 1, 0))
        # utils.vis3d.add_lines(torch.stack([db.Tv2w[:, :3, 3], db.Tv2w_origin[:, :3, 3]], dim=1), color=(0.1, 0.1, 0.1))
        # utils.vis3d.add_lines(points=db.Tv2w[:, :3, 3], color=(0.1, 0.1, 0.1))
        # utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2)[40::80, 40::80])
        # inputs = db.random_ray(None, 512)[0]
        # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        # utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.1, 0.1, 0.1))
        # inputs = db.random_ray(1, 10)[0]
        # rays_o, rays_d = inputs['rays_o'], inputs['rays_d']
        # utils.vis3d.add_lines(torch.stack([rays_o, rays_d + rays_o], dim=-2), color=(0.3, 0.3, 0.3))
        utils.vis3d.add_lines(points=[
            [-1., -1, -1],  # 0
            [-1., -1., 1],  # 1
            [-1., 1., -1.],  # 2
            [-1., 1., 1.],  # 3
            [1., -1., -1],  # 4
            [1, -1, 1],  # 5
            [1, 1, -1],  # 6
            [1, 1, 1],  # 7
        ],
            line_index=[[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])
    # db.batch_mode = False
    # print('batch_mode=False', utils.show_shape(db[0, 5]))
    # db.batch_mode = True
    # print('batch_mode=True', utils.show_shape(db[0, 5]))


def test():
    train_db = ColmapDataset(Path('~/data/NeRF/HyperNeRF/vrig-3dprinter/colmap').expanduser(), split='train')
    test_db = ColmapDataset(Path('~/data/NeRF/HyperNeRF/vrig-3dprinter/colmap').expanduser(), split='test')
    print(train_db)
    print(test_db)
    inputs, targets, infos = train_db[0]
    print('inputs:', utils.show_shape(inputs))
    print('targets:', utils.show_shape(targets))
    print('infos:', utils.show_shape(infos))
    plt.imshow(targets['images'].numpy())
    plt.axis('off')
    plt.show()

    with utils.vis3d:
        utils.vis3d.add_camera_poses(
            train_db.Tv2w,
            fovy=np.rad2deg(train_db.FoV[1].item()),
            aspect=train_db.aspect,
            color=(1, 0, 0),
            size=0.1
        )
        utils.vis3d.add_camera_poses(
            test_db.Tv2w,
            fovy=np.rad2deg(test_db.FoV[1].item()),
            aspect=test_db.aspect,
            color=(0, 1, 0),
            size=0.1
        )
        utils.vis3d.add_lines(points=[
            [-1., -1, -1],  # 0
            [-1., -1., 1],  # 1
            [-1., 1., -1.],  # 2
            [-1., 1., 1.],  # 3
            [1., -1., -1],  # 4
            [1, -1, 1],  # 5
            [1, 1, -1],  # 6
            [1, 1, 1],  # 7
        ],
            line_index=[[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])


if __name__ == '__main__':
    test()
