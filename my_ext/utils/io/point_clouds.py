from pathlib import Path

import numpy as np
from torch import Tensor

__all__ = ['load_point_clouds', 'save_point_clouds', 'point_clouds_extensions']

point_clouds_extensions = ['.xyz', '.pcd', '.obj', '.ply', '.ptx', '.pts']


def load_by_trimesh():
    pass


def load_by_ply():
    pass


def load_by_open3d():
    pass


def load_point_clouds(filepath, ext: str = None):
    import trimesh.exchange.ply
    filepath = Path(filepath)
    if ext is None:
        ext = filepath.suffix
    assert ext in point_clouds_extensions
    if ext == '.ply':
        with open(filepath, 'rb') as f:
            data = trimesh.exchange.ply.load_ply(f)
        vertices = data['vertices']
        return vertices
    else:
        raise NotImplementedError


def save_by_trimesh(filename: Path, vertices, colors=None, **kwargs):
    import trimesh.points
    if isinstance(vertices, Tensor):
        vertices = vertices.detach().cpu().numpy()
    if colors is not None:
        if isinstance(colors, Tensor):
            colors = colors.detach().cpu().numpy()
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
        if colors.shape[-1] == 3:
            colors = np.concatenate([colors, np.full_like(colors[:, :1], 255)], axis=-1)
    kwargs = {k: (v.detach().cpu().numpy() if isinstance(v, Tensor) else v) for k, v in kwargs.items() if v is not None}
    pcd = trimesh.points.PointCloud(vertices, colors, kwargs)
    pcd.export(filename)


def save_point_clouds(filepath, vertices, colors=None, ext: str = None, use='trimesh', **kwargs):
    filepath = Path(filepath)
    if ext is None:
        ext = filepath.suffix
    assert ext in point_clouds_extensions
    if use == 'trimesh':
        save_by_trimesh(filepath, vertices, colors=colors, **kwargs)
    else:
        raise NotImplementedError(f'can not use {use} to save point clouds')
    # if ext == '.ply':
    #     if normals is None:
    #         write_ply_point(filepath, vertices)
    #     else:
    #         write_ply_point_normal(filepath, vertices)
    # else:
    #     raise NotImplementedError(f"{ext} when save {filepath}")


def write_ply_point(name, vertices):
    if isinstance(vertices, Tensor):
        vertices = vertices.detach().cpu().numpy()
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    if isinstance(vertices, Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(normals, Tensor):
        normals = normals.detach().cpu().numpy()
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(
                vertices[ii, 3]) + " " + str(vertices[ii, 4]) + " " + str(vertices[ii, 5]) + "\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(
                normals[ii, 0]) + " " + str(normals[ii, 1]) + " " + str(normals[ii, 2]) + "\n")
    fout.close()
