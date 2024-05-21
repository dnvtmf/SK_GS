from typing import Union

import dearpygui.dearpygui as dpg
import numpy as np
from torch import Tensor
import cv2


class ImageViewer:

    def __init__(self, image=None, size=(100, 100), channels=3, pad=0, tag='image', **kwargs) -> None:
        self.pad = pad
        if image is None:
            image = np.ones((size[1], size[0], channels), dtype=np.float32)
        assert image.ndim == 3 and image.shape[-1] in [3, 4]
        self.size = (image.shape[1], image.shape[0])
        self.channels = channels
        assert self.channels in [3, 4]
        self._data = (image.astype(np.float32) / 255) if image.dtype == np.uint8 else image.astype(np.float32)
        self._origin_data = None
        self._can_dynamic_change = False
        self.pad = pad
        self.tag = tag
        with dpg.texture_registry(show=False) as self._registry_id:
            # self.registry_id = registry_id
            self._texture_id = dpg.add_raw_texture(
                self.width,
                self.height,
                default_value=self._data,  # noqa
                format=dpg.mvFormat_Float_rgba if self.channels == 4 else dpg.mvFormat_Float_rgb,
                tag=tag
            )
        W, H = self.size
        self._win_id = dpg.add_window(
            width=W + 2 * self.pad, height=H + 2 * self.pad, no_title_bar=True, no_scrollbar=True, **kwargs
        )
        self._img_id = dpg.add_image(self.tag, width=W, height=H, parent=self._win_id)

        with dpg.theme() as container_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, self.pad, self.pad, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(self._win_id, container_theme)

        self.resize_with_window()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_image: np.ndarray):
        assert new_image.shape == self._data.shape
        self._data[:] = (new_image / 255. if new_image.dtype == np.uint8 else new_image).astype(np.float32)

    @property
    def origin_data(self):
        if not self._can_dynamic_change:
            return self.data
        if self._origin_data is None:
            self._origin_data = self.data.copy()
        return self._origin_data

    @property
    def win_tag(self):
        return self._win_id

    @property
    def image_tag(self):
        return self._img_id

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def resize_with_window(self):
        def resize_handler(sender):
            H, W = dpg.get_item_height(self._win_id), dpg.get_item_width(self._win_id)
            self.resize(W - 2 * self.pad, H - 2 * self.pad)

        with dpg.item_handler_registry() as hr_id:
            dpg.add_item_resize_handler(callback=resize_handler)
        dpg.bind_item_handler_registry(self._win_id, hr_id)

    def resize(self, W: int = None, H: int = None, channels: int = None):
        W = self.width if W is None else W
        H = self.height if H is None else H
        channels = self.channels if channels is None else channels
        if (W, H) == self.size and channels == self.channels:
            return False
        assert self.channels in [3, 4]
        new_image = np.ones((H, W, channels), dtype=np.float32)
        min_H, min_W, min_c = min(H, self.height), min(W, self.width), min(channels, self.channels)
        new_image[:min_H, :min_W, :min_c] = self.data[:min_H, :min_W, :min_c]
        self._data = new_image
        if self._origin_data is not None:
            new_image = np.ones_like(self.data)
            new_image[:min_H, :min_W, :min_c] = self._origin_data[:min_H, :min_W, :min_c]
            self._origin_data = new_image
        self.channels = channels
        self.size = W, H

        # console.log(f'resize "{self.tag}": W={W}, H={H}')
        dpg.delete_item(self.tag)
        dpg.remove_alias(self.tag)
        dpg.hide_item(self._img_id)  # can not delete old image due to segmentation fault (core dumped)

        self._texture_id = dpg.add_raw_texture(
            W,
            H,
            default_value=self.data,  # noqa
            format=dpg.mvFormat_Float_rgba if self.channels == 4 else dpg.mvFormat_Float_rgb,
            tag=self.tag,
            parent=self._registry_id
        )
        self._img_id = dpg.add_image(self._texture_id, parent=self._win_id)
        dpg.configure_item(self._win_id, width=W + 2 * self.pad, height=H + 2 * self.pad)
        return True

    def update(self, image: Union[np.ndarray, Tensor], resize=False):
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        elif image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=-1)
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose((1, 2, 0))
        if resize:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        else:
            self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image
        self._origin_data = None

    def get_mouse_pos(self):
        x, y = dpg.get_mouse_pos(local=False)
        wx, wy = dpg.get_item_pos(self._win_id)
        ix, iy = dpg.get_item_pos(self._img_id)
        return int(x - wx - ix), int(y - wy - iy)

    def enable_dynamic_change(self, hover_callback=None):
        self._can_dynamic_change = True
        if hover_callback is None:
            return
        with dpg.item_handler_registry() as handler:
            dpg.add_item_hover_handler(callback=hover_callback)
        dpg.bind_item_handler_registry(self._img_id, handler)


class ImagesGUI:

    def __init__(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title='ImagesGUI', width=800, height=600)
        with dpg.window(tag='Primary Window'):
            img = ImageViewer(pos=(300, 100), pad=5, no_move=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window('Primary Window', True)
        dpg.start_dearpygui()
        dpg.destroy_context()


def test():
    dpg.create_context()
    dpg.create_viewport(title='Custom Title', width=800, height=600)
    with dpg.window(tag='Primary Window'):
        img = ImageViewer(pos=(300, 100), pad=5, no_move=False)
        # img.resize_with_window()
        with dpg.window():
            def callback(sender, app_data, user_data):
                img.data[:, :, :] = np.array(dpg.get_value(sender)) / 255

            dpg.add_color_picker([int(255) * x for x in img.data[0, 0]],
                tag='color_picker',
                no_side_preview=True,
                alpha_bar=True,
                width=200,
                callback=callback)
            dpg.add_input_int(label='W', default_value=img.width, tag='W')
            dpg.add_input_int(label='H', default_value=img.height, tag='H')
            dpg.add_button(label='Resize', callback=lambda *arg: img.resize(dpg.get_value('W'), dpg.get_value('H')))
            dpg.add_button(label='random image', callback=lambda *arg: img.update(np.random.rand(200, 200, 4)))
    with dpg.handler_registry():
        def show_pos(*args):
            print('local:', dpg.get_mouse_pos(local=True))
            print('golbal:', dpg.get_mouse_pos(local=False))
            print('win pos:', dpg.get_item_pos(img._win_id))
            print('img pos:', dpg.get_item_pos(img._img_id))
            print(
                'pos:',
                np.array(dpg.get_mouse_pos(local=False)) - np.array(dpg.get_item_pos(img._img_id)) -
                np.array(dpg.get_item_pos(img._win_id))
            )
            print()

        dpg.add_mouse_down_handler(callback=show_pos)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window('Primary Window', True)
    dpg.start_dearpygui()
    dpg.destroy_context()
