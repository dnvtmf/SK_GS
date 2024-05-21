import math
from copy import deepcopy
from typing import Union

import cv2
import numpy as np
import torch
from torch import Tensor

from my_ext.structures import Structure3D, Restoration
from my_ext.utils.torch_utils import show_shape
from my_ext.utils import geometry

__all__ = ['BoxList3D']


class BoxList3D(Structure3D):
    """
    This class represents a set of bounding boxes 3D.
    The bounding boxes are represented as a Nx? Tensor.

    mode:
        "cdr": center-dimensions-rotation_y: Nx7 [x, y, z, W, L, H, yaw]

        kitti": dimenstion-origin-rotation_y: Nx7  [H, W, L, x, y, z, rotation_y]
            (x, y, z) 底部中心点坐标，y轴指向地面, x-L, y-H, z-W

        "corners": Nx8x3 [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

        "xyz": Nx6 [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    MODE_CENTER_SIZE_YAW = 'cdr'
    MODE_CORNERS = 'corners'
    MODE_KITTI = 'kitti'

    def __init__(
        self, bbox: Union[Tensor, np.ndarray, list, None], mode="cdr",
        infos: dict = None,
        extra_fields: dict = None
    ):
        super().__init__(infos)
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        if mode not in [self.MODE_CENTER_SIZE_YAW, self.MODE_CORNERS, self.MODE_KITTI]:
            raise ValueError("mode should be 'cdr', 'corners'")

        if isinstance(bbox, np.ndarray):
            bbox = torch.from_numpy(bbox).float()
        elif isinstance(bbox, Tensor):
            bbox = bbox
        else:
            if not bbox:  # None, [], ()
                if mode == self.MODE_CENTER_SIZE_YAW or mode == self.MODE_KITTI:
                    bbox = torch.empty(0, 7, dtype=torch.float32, device=device)
                elif mode == self.MODE_CORNERS:
                    bbox = torch.empty(0, 8, 3, dtype=torch.float32, device=device)

            else:
                bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)

        self.bbox = bbox  # type: Tensor
        self.mode = mode
        self.extra_fields = {} if extra_fields is None else extra_fields

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data
        return self

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def clone(self):
        copy_ = BoxList3D(self.bbox.clone(), self.mode, deepcopy(self.infos), deepcopy(self.extra_fields))
        for k, v in self.extra_fields.items():
            if hasattr(v, 'copy'):
                copy_.add_field(k, getattr(v, 'copy')())
            else:
                copy_.add_field(k, deepcopy(v))
        return copy_

    def _convert_CSY_to_corners(self) -> Tensor:
        # corners_norm = np.stack(np.unravel_index(np.arange(2 ** 3), [2] * 3), axis=1)
        # corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # corners_norm = corners_norm - 0.5
        # corners = torch.from_numpy(corners_norm)  # type: Tensor
        corners = self.bbox.new_tensor([[
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]])
        corners = corners * self.bbox[:, None, 3:6]
        ang = self.bbox[:, 6]
        ang_sin = ang.sin()
        ang_cos = ang.cos()
        R = corners.new_zeros((ang.shape[0], 3, 3))
        R[:, 0, 0] = ang_cos
        R[:, 0, 1] = -ang_sin
        R[:, 1, 0] = ang_sin
        R[:, 1, 1] = ang_cos
        R[:, 2, 2] = 1
        points_box = corners @ R + self.bbox[:, None, :3]
        return points_box

    def _convert_KITTI_to_corners(self):
        corners = self.bbox.new_tensor([[
            [-0.5, 0, -0.5], [-0.5, -1, -0.5], [-0.5, 0, 0.5], [-0.5, -1, 0.5],
            [0.5, 0, -0.5], [0.5, -1, -0.5], [0.5, 0, 0.5], [0.5, -1, 0.5]]])
        corners = corners * self.bbox[:, None, (2, 0, 1)]
        ang = self.bbox[:, 6]
        ang_sin = ang.sin()
        ang_cos = ang.cos()
        R = corners.new_zeros((ang.shape[0], 3, 3))
        R[:, 0, 0] = ang_cos
        R[:, 0, 2] = -ang_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = ang_sin
        R[:, 2, 2] = ang_cos
        points_box = corners @ R + self.bbox[:, None, 3:6]
        return points_box

    def _convert_corners_to_CSY(self):
        center = self.bbox[:, :, :].mean(dim=1)
        width = torch.maximum(
            geometry.distance_point_to_line(self.bbox[:, 0, :2], self.bbox[:, 5, :2], self.bbox[:, 7, :2]),
            geometry.distance_point_to_line(self.bbox[:, 4, :2], self.bbox[:, 1, :2], self.bbox[:, 3, :2]),
        )
        length = torch.maximum(
            geometry.distance_point_to_line(self.bbox[:, 0, :2], self.bbox[:, 3, :2], self.bbox[:, 7, :2]),
            geometry.distance_point_to_line(self.bbox[:, 2, :2], self.bbox[:, 1, :2], self.bbox[:, 5, :2]),
        )
        height = self.bbox[:, :, 2].amax(dim=1) - self.bbox[:, :, 2].amin(dim=1)
        yaw = torch.atan2(self.bbox[:, 0, 1] - self.bbox[:, 4, 1], self.bbox[:, 4, 0] - self.bbox[:, 0, 0])
        return torch.cat([center, width[:, None], length[:, None], height[:, None], yaw[:, None]], dim=-1)

    def convert_(self, mode):
        if mode == self.mode:
            return self
        if self.mode == self.MODE_CENTER_SIZE_YAW and mode == self.MODE_CORNERS:
            self.bbox = self._convert_CSY_to_corners()
        elif self.mode == self.MODE_KITTI and mode == self.MODE_CORNERS:
            self.bbox = self._convert_KITTI_to_corners()
        elif self.mode == self.MODE_CORNERS and mode == self.MODE_CENTER_SIZE_YAW:
            self.bbox = self._convert_corners_to_CSY()
        else:
            raise NotImplementedError(f'Can not convert BoxList3D from {self.mode} to {mode}')
        self.mode = mode
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self.clone()
        if self.mode == 'cdr' and mode == 'corners':
            return BoxList3D(self._convert_CSY_to_corners(), mode=mode, infos=self.infos)
        elif self.mode == self.MODE_KITTI and mode == self.MODE_CORNERS:
            return BoxList3D(self._convert_KITTI_to_corners(), mode=mode, infos=self.infos)
        elif self.mode == self.MODE_CORNERS and mode == self.MODE_CENTER_SIZE_YAW:
            return BoxList3D(self._convert_corners_to_CSY(), mode=mode, infos=self.infos)
        else:
            raise NotImplementedError(f'Can not convert BoxList3D from {self.mode} to {mode}')

    def pan_zoom_(self, scaling=1., offset=0., *args, **kwargs):
        scaling = self.bbox.new_tensor(scaling)
        offset = self.bbox.new_tensor(offset)
        if self.mode == self.MODE_CENTER_SIZE_YAW:
            self.bbox[:, :3] = (self.bbox[:, :3] + offset) * scaling
            self.bbox[:, 3:6] = self.bbox[:, 3:6] * scaling
        else:
            raise NotImplementedError
        return self

    def flip_(self, x_axis=False, y_axis=False, z_axis=False, center=(0., 0, 0), *args, **kwargs):
        if x_axis:
            if self.mode == self.MODE_CENTER_SIZE_YAW:
                # self.bbox[:, 0].neg_()
                self.bbox[:, 0] = torch.add(-self.bbox[:, 0], center[0], alpha=2.)
                self.bbox[:, -1] = math.pi - self.bbox[:, -1]
            elif self.mode == self.MODE_CORNERS:
                self.bbox[:, :, 0] = torch.add(-self.bbox[:, :, 0], center[0], alpha=2.)
            else:
                raise NotImplementedError
        if y_axis:
            if self.mode == self.MODE_CENTER_SIZE_YAW:
                self.bbox[:, 1] = torch.add(-self.bbox[:, 1], center[1], alpha=2.)
                self.bbox[:, -1].neg_()
            elif self.mode == self.MODE_CORNERS:
                self.bbox[:, :, 1] = torch.add(-self.bbox[:, :, 1], center[1], alpha=2.)
            else:
                raise NotImplementedError
        if z_axis:
            if self.mode == self.MODE_CENTER_SIZE_YAW:
                self.bbox[:, 2] = torch.add(-self.bbox[:, 2], center[2], alpha=2.)
            elif self.mode == self.MODE_CORNERS:
                self.bbox[:, :, 2] = torch.add(-self.bbox[:, :, 2], center[2], alpha=2.)
            else:
                raise NotImplementedError
        return self

    def crop_(self, x, y, w, h, remove_area=-1, clamp=True, overlap=False, *args, **kwargs):
        return NotImplemented

    def crop(self, x, y, w, h, remove_area=-1, clamp=True, overlap=False, *args, **kwargs):
        return NotImplemented

    def pad_(self, padding, *args, **kwargs):
        return NotImplemented

    def pad(self, padding, *args, **kwargs):
        return NotImplemented

    # Tensor-like methods
    def to(self, device):
        bbox = BoxList3D(self.bbox.to(device), self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = getattr(v, "to")(device)
            bbox.add_field(k, v)
        bbox.add_info(**self.infos)
        return bbox

    def __getitem__(self, item):
        if self.bbox.numel() == 0:
            return self
        else:
            if isinstance(item, slice):
                bbox = BoxList3D(self.bbox[item], self.mode)
                for k, v in self.extra_fields.items():
                    bbox.add_field(k, v[item])
            else:
                if isinstance(item, int):
                    item = [item]
                bbox = BoxList3D(self.bbox[item] if len(item) else None, self.mode)
                for k, v in self.extra_fields.items():
                    bbox.add_field(k, v[item] if len(item) else None)
            bbox.add_info(**self.infos)
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def filter_out_range(self):
        return NotImplemented

    def remove_empty_(self, remove=True):
        return NotImplemented

    def copy_with_fields(self, fields):
        bbox = BoxList3D(self.bbox, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox.add_field(field, self.get_field(field))
        bbox.add_info(**self.infos)
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"num_boxes={len(self)}, "
        s += f"mode={self.mode}"
        if len(self.extra_fields) > 0:
            s += ", field: [" + ", ".join(self.extra_fields.keys()) + "]"
        if len(self.infos) > 0:
            s += f", infos: {show_shape(self.infos)}"
        s += ")"
        return s

    def affine_transform_(self, affine_matrix, scaling=None, rotate_z=None, flip=None, *args, **kwargs):
        if self.mode == self.MODE_CENTER_SIZE_YAW:
            """需要 只绕z轴的旋转，且在x和y方向的缩放尺度一致 """
            if scaling is not None and scaling[0] == scaling[1]:
                self.bbox[:, :3] = self.bbox[:, :3] @ affine_matrix[:3, :3] + affine_matrix[3, :3]  # xyz
                self.bbox[:, 3:6] = self.bbox[:, 3:6] * self.bbox.new_tensor(scaling)  # WLH
                if rotate_z is not None:
                    self.bbox[:, 6] += rotate_z
                if flip is not None:
                    if flip[0]:
                        self.bbox[:, 6] = math.pi - self.bbox[:, 6]
                    if flip[1]:
                        self.bbox[:, 6] = -self.bbox[:, 6]
            else:
                self.convert_(self.MODE_CORNERS)
                self.bbox = self.bbox @ affine_matrix[:3, :3] + affine_matrix[3, :3]  # xyz
        elif self.mode == self.MODE_CORNERS:
            self.bbox = self.bbox @ affine_matrix[:3, :3] + affine_matrix[3, :3]  # xyz
        else:
            raise NotImplementedError
        return self

    def restore(self, rt: Restoration):
        result = self
        for t in rt.get_transforms():
            result = getattr(result, t[0])(*t[1:-1], **t[-1])
        return result

    def vis(self, vis, line_color=(1, 0, 0), **kwargs):
        import open3d as o3d
        bbox = self.convert(self.MODE_CORNERS)
        lines_box = np.array([[0, 1], [0, 2], [1, 3], [2, 3],
                              [4, 5], [4, 6], [5, 7], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])

        for box in bbox.bbox:
            colors = np.array([line_color for j in range(len(lines_box))])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add(line_set)
        return

    def draw(self, img: np.ndarray, camera):
        from my_ext.structures import CameraIntrinsics
        assert isinstance(camera, CameraIntrinsics)
        p = self.convert(self.MODE_CORNERS).bbox
        xy = camera.camera_to_pixel(p)
        xy = xy.int().cpu().numpy()
        lines = [[4, 5], [5, 7], [6, 7], [4, 6], [2, 3], [3, 7], [2, 6], [0, 1], [0, 2], [1, 3], [0, 4], [1, 5]]
        for i in range(len(xy)):
            for j, (p1, p2) in enumerate(lines):
                img = cv2.line(img, tuple(xy[i, p1]), tuple(xy[i, p2]), (0, 255, 0) if j < 4 else (0, 0, 255))
            # img = cv2.fillConvexPoly(img, xy[i, (2, 3, 7, 6)], (0, 255, 0, 0))
        return img


def test_flip():
    from my_ext.utils.utils_3d.vis_3d import Vis3D
    box1 = BoxList3D([[1, 2, 3, 2, 4, 6, math.radians(30)]])
    # box1 = BoxList3D([[-1.99, -2.44, -2.55, 1.11, 2.22, 3.33, math.radians(60)]])
    box2 = box1.convert(BoxList3D.MODE_CORNERS)
    print()
    # center = (0, 0, 0)
    center = (-1, -1, -1)
    with Vis3D(width=1024, height=1024, lookat=[0, 0, 0], eye=[0, 0, 10], up=[0, 0, 1], field_of_view=120) as vis:
        box1.draw(vis, line_color=(0, 0, 0))
        box1.flip(x_axis=True, center=center).draw(vis, line_color=(1, 0, 0))
        box1.flip(y_axis=True, center=center).draw(vis, line_color=(0, 1, 0))
        box1.flip(z_axis=True, center=center).draw(vis, line_color=(0, 0, 1))
        box1.flip(x_axis=True, y_axis=True, center=center).draw(vis, line_color=(1, 1, 0))
        box2.vis(vis, line_color=(0, 0, 0))
        box2.flip(x_axis=True, center=center).draw(vis, line_color=(1, 0, 0))
        box2.flip(y_axis=True, center=center).draw(vis, line_color=(0, 1, 0))
        box2.flip(z_axis=True, center=center).draw(vis, line_color=(0, 0, 1))
        box2.flip(x_axis=True, y_axis=True, center=center).draw(vis, line_color=(1, 1, 0))


def test_affine():
    from my_ext.utils.utils_3d.vis_3d import Vis3D
    box1 = BoxList3D([[-2, -2, -2, 1, 2, 3, math.radians(30)]])
    # box1 = BoxList3D([[-1.99, -2.44, -2.55, 1.11, 2.22, 3.33, math.radians(60)]])
    box2 = box1.convert(BoxList3D.MODE_CORNERS)
    print()
    print('box1', box1, box1.bbox)
    print('box2', box2, box2.bbox)
    scaling = (1.1, 1.1, 1.3)
    rotate_z = -math.radians(30)
    with Vis3D(width=1024, height=1024, lookat=[0, 0, 0], eye=[0, 0, 10], up=[0, 0, 1], field_of_view=120) as vis:
        box1.vis(vis, line_color=(0, 0, 0))

        affine_matrix = box1.get_affine_matrix_3d((0, 0, rotate_z), scaling, flip=(False, False, False))
        box_ = box1.affine_transform(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=(False, False, False))
        box_.vis(vis, line_color=(0, 1, 1))
        print(box_.bbox)

        affine_matrix = box1.get_affine_matrix_3d((0, 0, rotate_z), scaling, flip=(True, False, False))
        box_ = box1.affine_transform(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=(True, False, False))
        box_.vis(vis, line_color=(1, 0, 0))
        print(box_.bbox)

        affine_matrix = box1.get_affine_matrix_3d((0, 0, rotate_z), scaling, flip=(False, True, False))
        box_ = box1.affine_transform(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=(False, True, False))
        box_.vis(vis, line_color=(0, 1, 0))
        print(box_.bbox)

        affine_matrix = box1.get_affine_matrix_3d((0, 0, rotate_z), scaling, flip=(False, False, True))
        box_ = box1.affine_transform(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=(False, False, True))
        box_.draw(vis, line_color=(0, 0, 1))
        print(box_.bbox)

        affine_matrix = box1.get_affine_matrix_3d((0, 0, rotate_z), scaling, flip=(True, True, False))
        box_ = box1.affine_transform(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=(True, True, False))
        box_.draw(vis, line_color=(1, 1, 0))
        print(box_.bbox)

    flip = (True, True, False)
    affine_matrix = box1.get_affine_matrix_3d(scaling=scaling, flip=flip)
    box1.affine_transform_(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=flip)
    box2.affine_transform_(affine_matrix, scaling=scaling, rotate_z=rotate_z, flip=flip)
    print('======= after afffine ==========')
    print(box1.bbox)
    print(box2.convert(box2.MODE_CENTER_SIZE_YAW).bbox)
    # print(box1.convert(box2.MODE_CORNERS).bbox)
    # print(box2.bbox)
