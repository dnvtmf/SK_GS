"""基于多线程和Open3D的三维数据显示"""
import os
import time
import warnings
import multiprocessing
from multiprocessing import Pipe, Process
from threading import Thread
from typing import Optional, Union, List, Tuple, TypeVar

import torch
import numpy as np
from torch import Tensor
import rich.traceback

rich.traceback.install()
try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    warnings.warn('Please install open3d')
    o3d = None
    gui = None
    rendering = None

__all__ = ['vis3d']


def to_open3d_type(value: Union[Tensor, np.ndarray]):
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    if value.dtype == np.int32:
        if value.ndim == 1:
            result = o3d.utility.IntVector(value)
        elif value.ndim == 2:
            if value.shape[-1] == 2:
                result = o3d.utility.Vector2iVector(value)
            elif value.shape[-1] == 3:
                result = o3d.utility.Vector3iVector(value)
            elif value.shape[-1] == 4:
                result = o3d.utility.Vector4iVector(value)
    elif value.dtype == np.float64:
        if value.ndim == 1:
            result = o3d.utility.DoubleVector(value)
        elif value.ndim == 2:
            if value.shape[-1] == 3:
                result = o3d.utility.Vector3dVector(value)
            elif value.shape[-1] == 2:
                result = o3d.utility.Vector2dVector(value)
        elif value.ndim == 3:
            if value.shape[-1] == 3:
                result = o3d.utility.Matrix3dVector(value)
    else:
        raise NotImplementedError(value.shape, value.dtype)
    return result


T_ARRAY = TypeVar('T_ARRAY', np.ndarray, Tensor)


def to_numpy(x: T_ARRAY) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


class Open3DVisualizationApp:

    def __init__(self, get_pipe, stop_event, keep_show, no_window):
        self.pipe = get_pipe  # type: multiprocessing.connection.Connection
        self.stop_event = stop_event  # type: multiprocessing.synchronize.Event
        self.keep_show = keep_show  # type: multiprocessing.synchronize.Event
        self.no_window = no_window  # type: multiprocessing.synchronize.Event

        self.window = gui.Application.instance.create_window('Open3D', 1360, 768)  # type: gui.Window
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.scene = rendering.Open3DScene(self.window.renderer)  # type: rendering.Open3DScene
        self.widget3d.scene = self.scene
        self.window.add_child(self.widget3d)

        em = self.window.theme.font_size
        self.panel = gui.Vert(1 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.window.add_child(self.panel)

        self.reset_camera_button = gui.Button('Reset Camera')
        self.reset_camera_button.set_on_clicked(self._reset_camera)
        self.panel.add_child(self.reset_camera_button)

        self.show_axis_option = gui.Checkbox('Show axis')
        self.show_axis_option.set_on_checked(lambda checked: self.scene.show_axes(checked))
        self.panel.add_child(self.show_axis_option)

        self.show_sky_option = gui.Checkbox('Show Sky')
        self.show_sky_option.set_on_checked(lambda checked: self.scene.show_skybox(checked))
        self.panel.add_child(self.show_sky_option)

        self.show_next_button = gui.Button('Next')
        self.show_next_button.set_on_clicked(self.keep_show.set)
        self.panel.add_child(self.show_next_button)

        self.close_button = gui.Button('Close')
        self.close_button.set_on_clicked(self._close_window)
        self.panel.add_child(self.close_button)

        collapse = gui.CollapsableVert("Other Options", 0.33 * em, gui.Margins(em, 0, 0, 0))
        collapse.set_is_open(False)

        color = gui.ColorEdit()
        color.color_value = gui.Color(0.5, 0.5, 0.5)
        color.set_on_value_changed(self._set_background_color)
        collapse.add_child(color)

        self.point_size = 5 * self.window.scaling
        self._add_change_point_size_button(collapse)
        self.line_width = 4 * self.window.scaling
        self._add_line_width_button(collapse)
        self.panel.add_child(collapse)

        self.is_done = False
        self.object_id = 0
        self.geometries_shaders = {}
        self.update_thread = Thread(target=self._update_gemotory)
        self.update_thread.start()

    def _update_gemotory(self):
        from rich.console import Console
        console = Console()
        while not self.is_done:
            if not self.pipe.poll(timeout=1):
                continue
            geometries = self.pipe.recv()
            mat = rendering.MaterialRecord()
            mat.point_size = self.point_size
            mat.shader = "defaultUnlit"
            mat.line_width = self.line_width

            def update():
                self.scene.clear_geometry()
                self.object_id = 0
                self.geometries_shaders.clear()
                for geometry_dict in geometries:
                    name = f"object{self.object_id}"
                    self.object_id += 1
                    try:
                        g = self.resolve_data(**geometry_dict)
                    except Exception:
                        console.print_exception()
                        self._close_window()
                    if isinstance(g, (o3d.geometry.LineSet, o3d.geometry.AxisAlignedBoundingBox)):
                        mat.shader = 'unlitLine'
                    self.geometries_shaders[name] = mat.shader
                    self.scene.add_geometry(name, g, mat)
                    self._reset_camera()

            gui.Application.instance.post_to_main_thread(self.window, update)
        return

    def _on_layout(self, layout_context):
        cR = self.window.content_rect
        em = layout_context.theme.font_size
        panel_width = 15 * em  # 15 ems wide
        self.widget3d.frame = gui.Rect(cR.x, cR.y, cR.width - panel_width, cR.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(), cR.y, panel_width, cR.height)

    def _on_close(self):
        self.stop_event.set()
        self.keep_show.set()
        self.no_window.set()
        self.is_done = True
        print('close window', flush=True)
        return True

    def _close_window(self):
        self.window.close()
        # print('_close_window window close')
        gui.Application.instance.quit()
        # print('_close_window app quit')

    def _reset_camera(self):
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())

    def resolve_data(self, open3d=None, function=None, **kwargs):
        if open3d is not None:
            return Open3DVisualizationApp.to_open3d_geometry(open3d, **kwargs)
        elif function is not None:
            assert hasattr(self, function)
            return getattr(self, function)(**kwargs)

    def add_points(self, points: np.ndarray, color=(0., 0., 0.)):
        g = o3d.geometry.PointCloud(to_open3d_type(points.reshape(-1, 3).astype(np.float64)))
        if isinstance(color, np.ndarray):
            if color.dtype == np.uint8 or color.dtype == np.int32 or color.dtype == np.int64:
                color = color.astype(np.float64) / 255.
            if color.ndim == 1:
                g.paint_uniform_color(color)
            else:
                g.colors = to_open3d_type(color.astype(np.float64))
        else:
            g.paint_uniform_color(color)
        return g

    def add_lines(
        self, lines: np.ndarray = None, points: np.ndarray = None, line_index: np.ndarray = None, color=(0., 0., 0.)
    ):
        """ 显示线段

        Args:
            lines: 由N个点构成的线段 shape: [..., N, 3]. Defaults to None.
            points: shape: [M, 3]. Defaults to None.
            line_index: shape: [LN, 2]. Defaults to None.
            color: 颜色 shape [LN, 3] or [3]. Defaults to (0., 0., 0.).
        Returns:
        """
        if lines is not None:
            N = lines.shape[-2]
            lines = lines.reshape(-1, N, 3)
            points = lines.reshape(-1, 3)
            line_index = np.stack([np.arange(N - 1), np.arange(N - 1) + 1], axis=-1)
            line_index = np.reshape(line_index[None, :, :] + np.arange(lines.shape[0])[:, None, None] * N, (-1, 2))
        else:
            assert points is not None
            M = points.shape[0]
            if line_index is None:
                line_index = np.stack([np.arange(M - 1), np.arange(1, M)], axis=-1)

        g = o3d.geometry.LineSet(to_open3d_type(points.astype(np.float64)), to_open3d_type(line_index.astype(np.int32)))
        if isinstance(color, np.ndarray):
            if color.dtype == np.uint8 or color.dtype == np.int32 or color.dtype == np.int64:
                color = color.astype(np.float64) / 255.
            if color.ndim == 1:
                g.paint_uniform_color(color)
            else:
                g.colors = to_open3d_type(color.astype(np.float64))
        else:
            g.paint_uniform_color(color)
        return g

    def add_camera_poses(self, Tv2w: np.ndarray, fovy=90., aspect=1., size=0.1, color=(0, 0, 0.)):
        fovy = np.deg2rad(fovy) * 0.5
        y = size * np.sin(fovy)
        x = y * aspect
        z = -size * np.cos(fovy)
        points = np.array([[0, 0, 0, 1], [x, y, z, 1], [x, -y, z, 1], [-x, -y, z, 1], [-x, y, z, 1]], dtype=np.float64)
        points = points @ Tv2w.swapaxes(-1, -2)
        points = points.reshape(-1, 4)[..., :3]
        n = points.shape[0] // 5
        indices = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
        indices = (indices[None, :, :] + np.arange(n, dtype=np.int32)[:, None, None] * 5)
        if isinstance(color, np.ndarray) and color.ndim == 2:
            assert color.shape == (n, 3)
            color = np.repeat(color, 8, axis=0)
        return self.add_lines(points=points, line_index=indices.reshape(-1, 2), color=color)

    def add_sphere(self, radius: float = 1., offset=(0., 0., 0.), color=(0., 0., 0.)):
        sphere = o3d.geometry.TriangleMesh().create_sphere(radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(offset)
        return sphere

    @staticmethod
    def to_open3d_geometry(class_name: str, data: List[Tuple[str, np.ndarray]]):
        obj = getattr(o3d.geometry, class_name)()
        for name, value in data:  # type: str, np.ndarray
            setattr(obj, name, to_open3d_type(value))
        return obj

    def _add_change_point_size_button(self, container):
        def check_point_size(new_value):
            self.point_size = new_value * self.window.scaling
            for name, shader in self.geometries_shaders.items():
                mat = rendering.MaterialRecord()
                mat.shader = shader
                mat.point_size = self.point_size
                mat.line_width = self.line_width
                self.scene.modify_geometry_material(name, mat)
            self.window.post_redraw()

        slider = gui.Slider(gui.Slider.INT)
        slider.int_value = int(self.point_size / self.window.scaling)
        slider.set_limits(1, 10)
        slider.set_on_value_changed(check_point_size)
        container.add_child(gui.Label('Point size'))
        container.add_child(slider)

    def _add_line_width_button(self, container):
        def check_linw_width(new_value):
            self.line_width = new_value * self.window.scaling
            for name, shader in self.geometries_shaders.items():
                mat = rendering.MaterialRecord()
                mat.shader = shader
                mat.point_size = self.point_size
                mat.line_width = self.line_width
                self.scene.modify_geometry_material(name, mat)
            self.window.post_redraw()

        slider = gui.Slider(gui.Slider.INT)
        slider.int_value = int(self.line_width / self.window.scaling)
        slider.set_limits(1, 10)
        slider.set_on_value_changed(check_linw_width)
        container.add_child(gui.Label('Line Width'))
        container.add_child(slider)

    def _set_background_color(self, color):
        # type: (gui.Color) -> None
        self.scene.set_background([color.red, color.green, color.blue, color.alpha])


class Vis3D:

    def __init__(self, enable=True):
        self.enable = enable and o3d is not None
        self.geometries = []
        if self.enable:
            self.out_pipe, self.in_pipe = Pipe()
            self.stop_event = multiprocessing.Event()
            self.keep_show = multiprocessing.Event()
            self.no_window = multiprocessing.Event()
        self._p = None  # type: Optional[Process]
        self._keep_show = False
        self._web_enabled = False

    def enable_web(self, ip='0.0.0.0', port=8888):
        if self._web_enabled:
            return
        is_on_local_machine = os.getenv('DISPLAY') is not None and len(os.environ['DISPLAY'].split(':')[0]) == 0
        if not is_on_local_machine:
            os.environ['WEBRTC_IP'] = ip
            os.environ['WEBRTC_PORT'] = str(port)
            o3d.visualization.webrtc_server.enable_webrtc()
            print(f'See open3d visualization: http://{ip}:{port}')
        self._web_enabled = True

    def add(self, *geometory):
        for g in geometory:
            self.geometries.append(self._to_numpy(g))

    def add_points(self, poionts: T_ARRAY, color=(0., 0., 0.)):
        self.geometries.append(dict(function='add_points', color=color, points=to_numpy(poionts)))

    def add_lines(self, lines: T_ARRAY = None, points: T_ARRAY = None, line_index: T_ARRAY = None, color=(0., 0., 0.)):
        data = dict(function='add_lines', color=color)
        if lines is not None:
            data['lines'] = to_numpy(lines)
        if points is not None:
            data['points'] = to_numpy(points)
        if line_index is not None:
            data['line_index'] = to_numpy(line_index)
        self.geometries.append(data)

    def add_camera_poses(
        self, Tv2w: T_ARRAY = None, Tw2v: T_ARRAY = None, fovy=90., aspect=1., size=0.1, color=(0, 0, 0.)
    ):
        """
        show camear poses

        Args:
            Tv2w: Union[np.ndarray, Tensor] Transform from view space to world shape [..., 4, 4]
            Tw2v: Union[np.ndarray, Tensor] Transform from world space to view shape [..., 4, 4]
            fovy:
            aspect: W/H
            size float: _description_. Defaults to 0.1.
            color (tuple, np.ndarray): _description_. Defaults to (0., 0., 0.).
        """
        assert Tv2w is not None or Tw2v is not None
        if Tv2w is not None:
            Tv2w = to_numpy(Tv2w)
        else:
            Tv2w = np.linalg.inv(to_numpy(Tw2v))
        self.geometries.append(
            dict(function='add_camera_poses', Tv2w=Tv2w, fovy=fovy, aspect=aspect, size=size, color=color)
        )

    def add_sphere(self, radius: float = 1., offset=(0., 0., 0.), color=(0., 0., 0.)):
        self.geometries.append({'function': 'add_sphere', 'radius': radius, 'offset': offset, 'color': color})

    def __enter__(self):
        from my_ext.distributed import is_main_process
        self.enable = self.enable and is_main_process()
        if self.enable:
            self.stop_event.clear()
            self._start_process()
            self.geometries.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        self.keep_show.clear()
        self._show()
        if self._keep_show:
            while not self.no_window.is_set():
                if self.keep_show.wait(1):
                    print('wait to show next', flush=True)
                    break
        self.keep_show.clear()
        self._keep_show = False

    def __call__(self, non_blocking=True):
        """阻塞调用"""
        self._keep_show = not non_blocking
        return self

    def _show(self):
        self.in_pipe.send(self.geometries)
        self.geometries.clear()

    def show(self, keep_show=True):
        from my_ext.distributed import is_main_process
        if not self.enable or not is_main_process:
            return
        self.stop_event.clear()
        self._start_process()
        self._show()
        if keep_show:
            while not self.no_window.is_set():
                if self.keep_show.wait(1):
                    print('wait to show next', flush=True)
                    break
        self.keep_show.clear()
        self._keep_show = False

    def _start_process(self):
        if self._p is not None and self._p.is_alive():
            if not self.no_window.is_set():
                # print('server is still running')
                return
            else:
                self._p.kill()  # window has closed
        self.no_window.clear()
        # self.enable_web()
        self._p = Process(target=self.run_server, args=(self.out_pipe, self.stop_event, self.keep_show, self.no_window))
        self._p.start()
        # print('start open3d window')

    def __del__(self):
        while self._p is not None and self._p.is_alive():
            print('wait to close open3d window')
            if self.stop_event.wait(1):
                break
        while self._p is not None and self._p.is_alive():
            print('kill the windows')
            self._p.kill()
        self.out_pipe.close()
        self.in_pipe.close()

    @staticmethod
    def run_server(*args, **kwargs):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
        app = gui.Application.instance  # type: gui.Application
        app.initialize()
        Open3DVisualizationApp(*args, **kwargs)

        app.run()
        # print('app exit', flush=True)

    @staticmethod
    def _to_numpy(obj) -> dict:
        """ Convert open3d Geometry to numpy array, 
        Args:
            obj (o3d.geometry.Geometry): 

        Returns:
            List[Tuple[str, Any]]: format: [(open3d, type_name), (attr1, value1), ...]
        """
        assert isinstance(obj, o3d.geometry.Geometry)
        data = {}
        obj_type = type(obj)
        type_name = obj_type.__name__
        data['open3d'] = type_name
        for k, v in obj_type.__dict__.items():
            if type(v) == property:
                array = np.asarray(getattr(obj, k))
                if array.size > 0:
                    data[k] = array
        return data


vis3d = Vis3D()


def _test():
    import numpy as np

    def make_point_cloud(npts, center, radius, colorize):
        pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        if colorize:
            colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
            cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    pc_rad = 1.0
    pc_nocolor = make_point_cloud(100, (0, -2, 0), pc_rad, False)
    pc_color = make_point_cloud(100, (3, -2, 0), pc_rad, True)
    r = 0.4
    sphere_unlit = o3d.geometry.TriangleMesh().create_sphere(r)
    sphere_unlit.translate((0, 1, 0))
    sphere_colored_unlit = o3d.geometry.TriangleMesh().create_sphere(r)
    sphere_colored_unlit.paint_uniform_color((1.0, 0.0, 0.0))
    sphere_colored_unlit.translate((2, 1, 0))
    sphere_lit = o3d.geometry.TriangleMesh().create_sphere(r)
    sphere_lit.compute_vertex_normals()
    sphere_lit.translate((4, 1, 0))
    sphere_colored_lit = o3d.geometry.TriangleMesh().create_sphere(r)
    sphere_colored_lit.compute_vertex_normals()
    sphere_colored_lit.paint_uniform_color((0.0, 1.0, 0.0))
    sphere_colored_lit.translate((6, 1, 0))
    big_bbox = o3d.geometry.AxisAlignedBoundingBox((-pc_rad, -3, -pc_rad), (6.0 + r, 1.0 + r, pc_rad))
    big_bbox.color = (0.0, 0.0, 0.0)
    sphere_bbox = sphere_unlit.get_axis_aligned_bounding_box()
    sphere_bbox.color = (1.0, 0.5, 0.0)
    lines = o3d.geometry.LineSet().create_from_axis_aligned_bounding_box(sphere_lit.get_axis_aligned_bounding_box())
    lines.paint_uniform_color((0.0, 1.0, 0.0))
    lines_colored = o3d.geometry.LineSet().create_from_axis_aligned_bounding_box(
        sphere_colored_lit.get_axis_aligned_bounding_box()
    )
    lines_colored.paint_uniform_color((0.0, 0.0, 1.0))

    with vis3d(non_blocking=True) as vis:
        # vis.enable_web()
        # vis.add(pc_nocolor, pc_color, sphere_unlit, sphere_colored_unlit, sphere_lit,
        #         sphere_colored_lit, big_bbox, sphere_bbox, lines, lines_colored)
        vis.add(pc_nocolor)
        vis.add(pc_color)
        vis.add(sphere_unlit)
        vis.add(sphere_colored_unlit)
        vis.add(sphere_lit)
        vis.add(sphere_colored_lit)
    print('===== before sleep =======')
    time.sleep(10)
    print('===== after sleep =======')
    with vis3d as vis:
        vis.add(big_bbox)
        vis.add(sphere_bbox)
        vis.add(lines)
        vis.add(lines_colored)
    print('=================end main=============================')


def test2():
    print(to_numpy(1.))
    print(to_numpy((1., 2.)))
    print(to_numpy([1., 2, 3.]))
    print(to_numpy(np.zeros(4)))
    print(to_numpy(torch.zeros(5)))
    poses = np.array([[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1.]],
                      [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1.]]],
        dtype=np.float32)

    with vis3d as vis:
        vis.add_camera_poses(Tv2w=poses, color=np.array([[255, 0, 0], [0, 255, 0]]))
        vis.add_lines(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]))


if __name__ == '__main__':
    # _test()
    test2()
