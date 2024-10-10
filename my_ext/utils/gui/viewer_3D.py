import math
from typing import Union, Callable

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from torch import Tensor

from .image_viewer import ImageViewer
from ..lazy_import import LazyImport

ops_3d = LazyImport('ops_3d', globals(), 'my_ext.ops_3d')


class Viewer3D(ImageViewer):

    def __init__(self, renderer: Callable, size=(100, 100), pad=0, tag='3d', no_resize=True, no_move=True, **kwargs):
        super().__init__(size=size, pad=pad, tag=tag, no_resize=no_resize, no_move=no_move, **kwargs)

        self.renderer = renderer
        self.fovy = math.radians(60.)
        self.Tv2s = ops_3d.camera_intrinsics(size=size, fovy=self.fovy)
        self.Ts2v = ops_3d.camera_intrinsics(size=size, fovy=self.fovy, inv=True)

        self.up = torch.tensor([0, 1., 0.])
        self.eye = torch.tensor([0., 0., 2.0])
        self.at = torch.tensor([0., 0., 0.])
        #
        self._last_mouse_pos = None
        self._last_mouse_idx = None
        self.rate_rotate = self.fovy / self.height  # 旋转速度
        self.rate_translate = 1.  # 平移速度
        self.need_update = True

    def resize(self, W: int = None, H: int = None, channels: int = None):
        if super().resize(W, H, channels):
            self.need_update = True

    def callback_mouse_down(self, sender, app_data):
        # if dpg.is_item_hovered(self._img_id):
        #     self._last_mouse_pos = self.get_mouse_pos()
        #     self._last_mouse_idx = app_data[0]
        #     print(sender, app_data, self._last_mouse_pos)
        # else:
        #     self._last_mouse_pos = None
        #     self._last_mouse_idx = None
        pass

    def callback_mouse_release(self, sender, app_data):
        self._last_mouse_pos = None
        self._last_mouse_idx = None

    def callback_mouse_wheel(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        self.scale(app_data)

    def callback_mouse_drag(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        if app_data[0] == dpg.mvMouseButton_Left:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.rotate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        elif app_data[0] == dpg.mvMouseButton_Right:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.translate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        self._last_mouse_pos = self.get_mouse_pos()
        self._last_mouse_idx = app_data[0]

    def rotate(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return
        radiu = (self.eye - self.at).norm()
        dir_vec = ops_3d.normalize(self.eye - self.at)
        right_vec = ops_3d.normalize(torch.linalg.cross(self.up, dir_vec), dim=-1)
        theta = -dy * self.rate_rotate
        dir_vec = ops_3d.quaternion.xfm(dir_vec, ops_3d.quaternion.from_rotate(right_vec, right_vec.new_tensor(theta)))

        right_vec = ops_3d.normalize(torch.linalg.cross(self.up, dir_vec), dim=-1)
        up_vec = torch.linalg.cross(dir_vec, right_vec)
        theta = -dx * self.rate_rotate
        dir_vec = ops_3d.quaternion.xfm(dir_vec, ops_3d.quaternion.from_rotate(up_vec, up_vec.new_tensor(float(theta))))
        self.eye = self.at + ops_3d.normalize(dir_vec) * radiu
        self.up = up_vec
        self.need_update = True

    def translate(self, dx: float, dy: float):
        """在垂直于视线方向进行平移, 即在view space进行平移"""
        if dx == 0 and dy == 0:
            return
        Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
        p1 = ops_3d.xfm(ops_3d.xfm(self.at, Tw2v), self.Tv2s)

        p2 = p1.clone()
        p2[0] += dx * p1[2]
        p2[1] += dy * p1[2]
        Tv2w = ops_3d.look_at(self.eye, self.at, self.up, inv=True)
        p1 = ops_3d.xfm(ops_3d.xfm(p1, self.Ts2v), Tv2w)
        p2 = ops_3d.xfm(ops_3d.xfm(p2, self.Ts2v), Tv2w)
        delta = (p1 - p2)[:3] * self.rate_translate
        self.at += delta
        self.eye += delta
        self.need_update = True

    def scale(self, delta=0.0):
        self.eye = self.at + (self.eye - self.at) * 1.1 ** (-delta)
        self.need_update = True

    def update(self, image: Union[np.ndarray, Tensor] = None, resize=False):
        if image is None and not self.need_update:
            return
        self.need_update = False
        if image is None:
            Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
            image = self.renderer(Tw2v, self.fovy, self.size)
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose(1, 2, 0)
        if resize:
            image = cv2.resize(image, self.size)
        self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image

    def set_fovy(self, fovy=60.):
        self.fovy = math.radians(fovy)
        self.Tv2s = ops_3d.camera_intrinsics(size=self.size, fovy=self.fovy)
        self.Ts2v = ops_3d.camera_intrinsics(size=self.size, fovy=self.fovy, inv=True)
        self.need_update = True

    def set_pose(self, eye=None, at=None, up=None, Tw2v=None, Tv2w=None):
        if Tv2w is None and Tw2v is not None:
            Tv2w = Tw2v.inverse()
        if Tv2w is not None:
            Tv2w = Tv2w.view(-1, 4, 4)[0].to(self.eye.device)
            eye, at, up = ops_3d.look_at_get(Tv2w)
        if eye is not None:
            self.eye = eye
        if at is not None:
            self.at = at
        if up is not None:
            self.up = up
        self.need_update = True

    def set_need_update(self, need_update=True):
        self.need_update = need_update

    def build_gui_camera(self):
        with dpg.group(horizontal=True):
            dpg.add_text('fovy')
            dpg.add_slider_float(
                min_value=15.,
                max_value=180.,
                default_value=math.degrees(self.fovy),
                callback=lambda *args: self.set_fovy(dpg.get_value('set_fovy')),
                tag='set_fovy'
            )
        with dpg.group():
            item_width = 50
            with dpg.group(horizontal=True):
                dpg.add_text('eye')
                dpg.add_input_float(tag='eye_x', step=0, width=item_width)
                dpg.add_input_float(tag='eye_y', step=0, width=item_width)
                dpg.add_input_float(tag='eye_z', step=0, width=item_width)
            with dpg.group(horizontal=True):
                dpg.add_text('at ')
                dpg.add_input_float(tag='at_x', step=0, width=item_width)
                dpg.add_input_float(tag='at_y', step=0, width=item_width)
                dpg.add_input_float(tag='at_z', step=0, width=item_width)

            def change_eye(*args):
                print('change camera position', args)
                self.eye = self.eye.new_tensor([dpg.get_value(item) for item in ['eye_x', 'eye_y', 'eye_z']])
                self.at = self.at.new_tensor([dpg.get_value(item) for item in ['at_x', 'at_y', 'at_z']])
                self.need_update = True

            def to_camera_pos(campos, up):
                def callback():
                    r = (self.eye - self.at).norm()
                    eye = self.eye.new_tensor(campos)
                    self.eye = eye / eye.norm(keepdim=True) * r + self.at
                    self.up = self.up.new_tensor(up)
                    self.set_need_update()

                return callback

            with dpg.group(horizontal=True):
                dpg.add_button(label='change', callback=change_eye)
                dpg.add_button(label='+X', callback=to_camera_pos((1, 0, 0), (0, 1, 0)))
                dpg.add_button(label='-X', callback=to_camera_pos((-1, 0, 0), (0, 1, 0)))
                dpg.add_button(label='+Y', callback=to_camera_pos((0, 1, 0), (0, 0, 1)))
                dpg.add_button(label='-Y', callback=to_camera_pos((0, -1, 0), (0, 0, 1)))
                dpg.add_button(label='+Z', callback=to_camera_pos((0, 0, 1), (0, 1, 0)))
                dpg.add_button(label='-Z', callback=to_camera_pos((0, 0, -1), (0, 1, 0)))

    def update_gui_camera(self):
        if self.need_update:
            dpg.set_value('eye_x', self.eye[0].item())
            dpg.set_value('eye_y', self.eye[1].item())
            dpg.set_value('eye_z', self.eye[2].item())
            dpg.set_value('at_x', self.at[0].item())
            dpg.set_value('at_y', self.at[1].item())
            dpg.set_value('at_z', self.at[2].item())


def simple_3d_viewer(rendering, size=(400, 400)):
    dpg.create_context()
    dpg.create_viewport(title='Custom Title')
    with dpg.window(tag='Primary Window'):
        img = Viewer3D(rendering, size=size, no_resize=False, no_move=True)
        with dpg.window(tag='control', width=256):
            dpg.add_text(tag='fps')
            img.build_gui_camera()

    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(callback=img.callback_mouse_drag)
        dpg.add_mouse_wheel_handler(callback=img.callback_mouse_wheel)
        dpg.add_mouse_release_handler(callback=img.callback_mouse_release)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window('Primary Window', True)
    # dpg.start_dearpygui()
    last_size = None
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        img.update_gui_camera()
        img.update()
        now_size = dpg.get_item_width(img._win_id), dpg.get_item_height(img._win_id)
        if last_size != now_size:
            dpg.configure_item('control', pos=(dpg.get_item_width(img._win_id), 0))
            dpg.set_viewport_width(dpg.get_item_width(img._win_id) + dpg.get_item_width('control'))
            dpg.set_viewport_height(dpg.get_item_height(img._win_id))
            last_size = now_size
        dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")
    dpg.destroy_context()
