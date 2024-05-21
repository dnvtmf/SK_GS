from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from torch import Tensor

__all__ = ['plt3D']


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


class plt3D:

    def __init__(self, figsize=None) -> None:
        self.fig = plt.figure(figsize=figsize)
        self.ax = Axes3D(self.fig, auto_add_to_figure=False)
        self.fig.add_axes(self.ax)
        self.ax: Axes3D

    def draw_3d_coordinate_system(self, mutation_scale=10, text_size=10):
        """绘制3D坐标系"""
        ax = self.ax
        # ax.set_axis_off()

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        zticks = ax.get_zticks()
        xmin, xmax = xticks[0], xticks[-1]
        ymin, ymax = yticks[0], yticks[-1]
        zmin, zmax = zticks[0], zticks[-1]
        x = 0 if xmin <= 0 <= xmax else xmin
        y = 0 if ymin <= 0 <= ymax else ymin
        z = 0 if zmin <= 0 <= zmax else zmin
        ax.arrow3D(xmin, y, z, xmax - xmin, 0, 0, mutation_scale=mutation_scale, ec='black', fc='black')
        ax.arrow3D(x, ymin, z, 0, ymax - ymin, 0, mutation_scale=mutation_scale, ec='black', fc='black')
        ax.arrow3D(x, y, zmin, 0, 0, zmax - zmin, mutation_scale=mutation_scale, ec='black', fc='black')
        ax.text3D(xmax, y, z, "x", size=text_size)
        ax.text3D(x, ymax, z, "y", size=text_size)
        ax.text3D(x, y, zmax, "z", size=text_size)

        ticks = []
        tick_len = (zticks[1] - zticks[0]) * 0.2
        xmin, xmax = ax.get_xbound()
        for xtick in xticks:
            if xmin <= xtick <= xmax:
                ticks.append([xtick, y, z])
                ticks.append([xtick, y, z + tick_len])
                ticks.append([np.nan, np.nan, np.nan])
        tick_len = (xticks[1] - xticks[0]) * 0.2

        ymin, ymax = ax.get_ybound()
        for ytick in yticks:
            if ymin <= ytick <= ymax:
                ticks.append([x, ytick, z])
                ticks.append([x + tick_len, ytick, z])
                ticks.append([np.nan, np.nan, np.nan])
        tick_len = (yticks[1] - yticks[0]) * 0.2
        zmin, zmax = ax.get_zbound()
        for ztick in zticks:
            if zmin <= ztick <= zmax:
                ticks.append([x, y, ztick])
                ticks.append([x, y + tick_len, ztick])
                ticks.append([np.nan, np.nan, np.nan])

        ax.plot(*np.array(ticks).T, color='black')
        print(ax.get_xticks())
        print(ax.get_yticks())
        print(ax.get_zticks())
        return

    def example(self):

        def lorenz(xyz, *, s=10, r=28, b=2.667):
            x, y, z = xyz
            x_dot = s * (y - x)
            y_dot = r * x - y - x * z
            z_dot = x * y - b * z
            return np.array([x_dot, y_dot, z_dot])

        dt = 0.01
        num_steps = 10000

        xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
        xyzs[0] = (0., 1., 1.05)  # Set initial values
        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(num_steps):
            xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

        self.ax.plot(*xyzs.T, lw=0.5)
        self.draw_3d_coordinate_system()

        self.show(True)
        return

    def update_view(self, num):
        azim = 360 * (num / 100)
        self.ax.view_init(elev=self.ax.elev, azim=azim)

    def show(self, auto_rotate=False):
        ani = animation.FuncAnimation(self.fig, self.update_view, np.arange(0, 100)) if auto_rotate else None
        plt.show()
        return ani

    def draw_camera_pose(self, poses: Union[np.ndarray, Tensor]):
        if isinstance(poses, Tensor):
            poses = poses.detach().cpu().numpy()
        ax = self.ax
        ax.scatter(poses[..., 0, 3], poses[..., 1, 3], poses[..., 2, 3])


if __name__ == '__main__':
    plt3D().example()