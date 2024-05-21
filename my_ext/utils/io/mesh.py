from typing import List, Union, Dict, Tuple
import os
from pathlib import Path
import struct

import numpy as np
import torch
from torch import Tensor

from .material import load_mtl, MTL

__all__ = [
    'load_mesh', 'load_obj', 'load_ply', 'save_mesh', 'save_off', 'save_obj', 'save_ply', 'save_glb', 'mesh_extensions',
    'parse_ply'
]

mesh_extensions = ['.obj', '.off', '.ply']


def load_obj(filename: Path, mtl: Union[bool, List[MTL], str] = True):
    """ load a .obj file

    Args:
        filename: The file path of .obj
        mtl: The option of materials.
            - True (default): load materials and delete unused MTL
            - False: Unload materials
            - List[MTL]: use given materials
            - 'keep': load materials but keep unused MTL

    Returns:
        dict: include vertices, faces, ..., materials
    """
    # TODO: use c++ to read
    # TODO: delay load mtl
    filename = Path(filename)
    texture_dir = filename.parent

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    all_materials = []
    if isinstance(mtl, (list, tuple)):
        assert all(isinstance(m, MTL) for m in mtl)
        all_materials = mtl
    elif mtl:
        assert mtl is True or mtl == 'keep'
        # all_materials.append(MTL('_default_mat'))
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'mtllib':
                # Read in entire material library
                all_materials += load_mtl(os.path.join(texture_dir, line.split()[1]))
    else:
        assert mtl is False

    # load vertices
    vertices, texcoords, normals = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'v':  # vertices
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':  # vertex texture
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':  # vertex normal
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = -1
    # used_materials = [] if mtl is True else [mat.name for mat in all_materials]
    mat_names = [mat.name for mat in all_materials]
    f_pos, f_tex, f_nrm, f_mat = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        prefix = line.split()[0].lower()
        if prefix == 'usemtl' and mtl is not False:  # Track used materials
            mat_name = line.split()[1]
            # if mat_name not in used_materials:
            #     if not any(mat.name == mat_name for mat in all_materials):
            #         all_materials.append(MTL(mat_name))
            #     used_materials.append(mat_name)
            # activeMatIdx = used_materials.index(mat_name)
            assert mat_name in mat_names
            activeMatIdx = mat_names.index(mat_name)
        elif prefix == 'f':  # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
            for i in range(nv - 2):  # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if len(vv) > 1 and vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if len(vv) > 2 and vv[2] != "" else -1
                f_mat.append(activeMatIdx)
                f_pos.append([v0, v1, v2])
                f_tex.append([t0, t1, t2])
                f_nrm.append([n0, n1, n2])
    assert len(f_tex) == len(f_pos) and len(f_nrm) == len(f_pos) and len(f_tex) == len(f_mat)

    outputs = {}
    outputs['v_pos'] = torch.tensor(vertices, dtype=torch.float32)
    outputs['f_pos'] = torch.tensor(f_pos, dtype=torch.int64)

    if activeMatIdx != -1:
        outputs['f_tex'] = torch.tensor(f_tex, dtype=torch.int64)
        if torch.any(outputs['f_tex'] < 0):
            texcoords.append([0.5, 0.5])
            outputs['f_tex'][outputs['f_tex'] < 0] = len(texcoords) - 1
        outputs['v_tex'] = torch.tensor(texcoords, dtype=torch.float32)
    if len(normals) > 0:
        outputs['v_nrm'] = torch.tensor(normals, dtype=torch.float32)
        outputs['f_nrm'] = torch.tensor(f_nrm, dtype=torch.int64)
        assert 0 <= outputs['f_nrm'].min() and outputs['f_nrm'].max() < len(outputs['v_nrm'])
    if mtl is not False:
        f_mat = torch.tensor(f_mat, dtype=torch.int64)
        if torch.any(f_mat.eq(-1)):
            all_materials.append(MTL('_none'))
            # f_mat[f_mat < 0] = len(used_materials)
            # used_materials.append('_none')
            f_mat[f_mat < 0] = len(all_materials) - 1
        if mtl == 'keep':
            outputs['materials'] = all_materials
            outputs['f_mat'] = f_mat
        else:
            used, inverse_indices = torch.unique(f_mat, return_inverse=True)
            outputs['f_mat'] = inverse_indices
            outputs['materials'] = [all_materials[i] for i in used]
            # for j, i in enumerate(used):
            #     mat_name = used_materials[i]
            #     # print(mat_name, f_mat.eq(i).sum(), outputs['f_mat'].eq(j).sum())
            #     is_checked = False
            #     for mat in all_materials:
            #         if mat.name == mat_name:
            #             outputs['materials'].append(mat)
            #             is_checked = True
            #             break
            #     assert is_checked
        # print('num materials:', len(outputs['materials']))
    return outputs


_ply_dtype_map = {
    'char': ('b', int, np.int8),
    'int8': ('b', int, np.int8),
    'uchar': ('B', int, np.uint8),
    'uint8': ('B', int, np.uint8),
    'short': ('h', int, np.int16),
    'int16': ('h', int, np.int16),
    'ushort': ('H', int, np.uint16),
    'uint16': ('H', int, np.uint16),
    'int': ('i', int, np.int32),
    'int32': ('i', int, np.int32),
    'uint': ('I', int, np.uint32),
    'uint32': ('I', int, np.uint32),
    'float': ('f', float, np.float32),
    'float32': ('f', float, np.float32),
    'double': ('d', float, np.float64),
    'float64': ('d', float, np.float64),
}  # type: Dict[str, Tuple[str, type(int), np.dtype]]


def _ply_get_format(dtypes, fmt: str):
    if fmt == 'binary_little_endian':
        fmt = '<'
    elif fmt == 'binary_big_endian':
        fmt = '>'
    else:
        assert fmt == 'ascii'
        return None
    for x in dtypes:
        fmt += _ply_dtype_map[x][0]
    return fmt, struct.calcsize(fmt)


def parse_ply(filename):
    """ parse a ply file
    reference: http://gamma.cs.unc.edu/POWERPLANT/papers/ply.pdf
    """
    # TODO: implement use C++
    error_msg = f"can not load ply file {filename}"
    with open(filename, 'rb') as f:
        ##### load header
        line_no = 0
        ply_format = None
        data = []
        while True:
            items = f.readline().decode('ascii').split()
            # print(items)
            if line_no == 0:
                assert len(items) == 1 and items[0] == 'ply', error_msg
            elif len(items) == 0 or items[0] == 'comment':
                continue
            elif items[0] == 'format':
                assert len(items) == 3 and items[2] == '1.0', error_msg
                ply_format = items[1]
                assert ply_format in ['ascii', 'binary_little_endian', 'binary_big_endian']
            elif items[0] == 'end_header':
                assert len(items) == 1, error_msg
                break
            elif items[0] == 'element':
                assert len(items) == 3, error_msg
                data.append({
                    'name': items[1],
                    'num': int(items[2]),
                    'dtypes': [],
                    'names': [],
                    'all_scalar': True,
                    'data': [],
                })
            elif items[0] == 'property':
                if items[1] == 'list':
                    assert len(items) == 5, error_msg
                    num_dtype = items[2]
                    data_type = items[3]
                    assert len(data) > 0
                    data[-1]['names'].append(items[-1])
                    data[-1]['dtypes'].append((num_dtype, data_type))
                    data[-1]['all_scalar'] = False
                else:
                    assert len(items) == 3, error_msg
                    data_type = items[1]
                    assert len(data) > 0
                    data[-1]['names'].append(items[-1])
                    data[-1]['dtypes'].append(data_type)

                assert data_type in [
                    'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double', 'int8', 'uint8', 'int16',
                    'uint16', 'int32', 'uint32', 'float32', 'float64'
                ], error_msg
            else:
                raise NotImplementedError('Undealed header:', items)
            line_no += 1
        assert ply_format is not None, error_msg
        ########## load context
        for elem in data:
            num = len(elem['names'])
            elem['data'] = [[] for _ in range(num)]
            if ply_format == 'ascii':
                for line_no in range(elem['num']):
                    values = f.readline().split()
                    row = []
                    i = 0
                    for dtype in elem['dtypes']:
                        if isinstance(dtype, str):
                            row.append(_ply_dtype_map[dtype][1](values[i].decode('ascii')))
                            i += 1
                        else:
                            num = int(values[i].decode('ascii'))
                            i += 1
                            row.append((_ply_dtype_map[dtype[1]][1](values[i + j].decode('ascii')) for j in range(num)))
                            i += num
                    assert len(row) == num
                    for i in range(num):
                        elem['data'][i].append(row[i])
                continue
            if elem['all_scalar']:
                x_formant, length = _ply_get_format(elem['dtypes'], ply_format)
                x_data = f.read(length * elem['num'])
                assert len(x_data) == length * elem['num']
                elem['data'] = list(zip(*struct.iter_unpack(x_formant, x_data)))
            else:
                fmts = []
                dtypes = []
                for dtype in elem['dtypes']:
                    if isinstance(dtype, str):
                        dtypes.append(dtype)
                    else:
                        dtypes.append(dtype[0])
                        fmts.append(_ply_get_format(dtypes, ply_format))
                        fmts.append(_ply_get_format([dtype[1]], ply_format))
                        dtypes.clear()
                if len(dtypes) > 0:
                    fmts.append(_ply_get_format(dtypes, ply_format))
                for line in range(elem['num']):
                    row = []
                    for i, (fmt, length) in enumerate(fmts):
                        if i % 2 == 0:
                            row.extend(struct.unpack_from(fmt, f.read(length)))
                        else:
                            cnt = row.pop(-1)
                            row.append(struct.unpack_from(f"{fmt[0]}{cnt}{fmt[1]}", f.read(length * cnt)))
                    for i in range(num):
                        elem['data'][i].append(row[i])
            # convert to numpy
            for i in range(num):
                if isinstance(elem['dtypes'][i], str):
                    dtype = elem['dtypes'][i]
                    elem['data'][i] = np.array(elem['data'][i], dtype=_ply_dtype_map[dtype][2])
                else:
                    len_data0 = len(elem['data'][i][0])
                    if all(len(x) == len_data0 for x in elem['data'][i]):
                        dtype = elem['dtypes'][i][1]
                        elem['data'][i] = np.array(elem['data'][i], dtype=_ply_dtype_map[dtype][2])
    return data


def load_ply(filename: Path):
    # parse to mesh format
    mesh_data = parse_ply(filename)

    def _to_array(element: dict, names: list, *extra, axis=-1):
        for name in names:
            assert name in element['names']
        indices = [element['names'].index(name) for name in names]
        indices.extend(element['names'].index(name) for name in extra if name in element['names'])
        dtype = element['dtypes'][indices[0]]
        for index in indices:
            assert element['dtypes'][index] == dtype
        return np.stack([element['data'][i] for i in indices], axis=axis)

    data = {}
    for elem in mesh_data:
        if elem['name'] == 'vertex':
            data['v_pos'] = _to_array(elem, ['x', 'y', 'z'])
            if 'nx' in elem['names']:
                data['v_nrm'] = _to_array(elem, ['nx', 'ny', 'nz'])
            if 'red' in elem['names']:
                data['v_clr'] = _to_array(elem, ['red', 'green', 'blue'], 'alpha')
            for name in elem['names']:
                if name not in ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', 'alpha']:
                    print(f'ply undealed atte {name} for vertex')
        elif elem['name'] == 'face':
            f_pos = []
            faces = elem['data'][elem['names'].index('vertex_indices')]
            # print(type(faces), faces.shape)
            if isinstance(faces, np.ndarray):
                if faces.shape[1] == 3:
                    data['f_pos'] = faces
                else:
                    for i in range(1, faces.shape[1] - 1):
                        f_pos.append(np.stack([faces[:, 0], faces[:, i], faces[:, i + 1]], axis=-1))
                    data['f_pos'] = np.stack(f_pos, axis=1).reshape(-1, 3)
            else:
                for ploy in faces:
                    for i in range(1, len(ploy) - 1):
                        f_pos.append((ploy[0], ploy[i], ploy[i + 1]))
                dtype = elem['dtypes'][elem['names'].index('vertex_indices')][1]
                data['f_pos'] = np.array(f_pos, dtype=_ply_dtype_map[dtype][2])
            for name in elem['names']:
                if name != 'vertex_indices':
                    print('ply undealed element', name)
        else:
            print('ply undealed element', elem['name'])
    data = {k: torch.from_numpy(v) for k, v in data.items()}
    return data


def load_mesh(filepath, ext: str = None, loader='open3d', **kwargs):
    filepath = Path(filepath)
    if ext is None:
        ext = filepath.suffix
        assert ext in mesh_extensions
    if loader == 'open3d':
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(filepath.as_posix())  # type: o3d.geometry.TriangleMesh
        return {
            'v_pos': torch.from_numpy(np.asarray(mesh.vertices)).float(),
            'f_pos': torch.from_numpy(np.asarray(mesh.triangles)).int(),
            'v_nrm': torch.from_numpy(np.asarray(mesh.vertex_normals)).float(),
            'v_clr': torch.from_numpy(np.asarray(mesh.vertex_colors)).float(),
        }
    elif loader == 'trimesh':
        import trimesh
        return trimesh.load(filepath, force='mesh')
    else:
        if ext == '.ply':
            return load_ply(filepath)
        elif ext == '.obj':
            return load_obj(filepath, **kwargs)
        else:
            raise NotImplementedError(f'cat not load mesh with extension {ext}')


############## save
def save_mesh(filepath, *args, ext: str = None):
    filepath = Path(filepath)
    if ext is None:
        ext = filepath.suffix
    assert ext in mesh_extensions, f'save mesh to {filepath}, unsupported extension {ext}'
    filepath = filepath.with_suffix(ext)
    if len(args) == 1:
        assert len(args[0]) == 2
        vertices, triangles = args[0]
    else:
        assert len(args) == 2
        vertices, triangles = args
    if ext == '.ply':
        save_ply(filepath, (vertices, triangles))
    elif ext == '.obj':
        save_obj(filepath, vertices, triangles)
    elif ext == '.off':
        save_off(filepath, vertices, triangles)
    else:
        raise NotImplementedError(ext)


def save_ply(file, data):
    import trimesh
    if not isinstance(data, trimesh.Trimesh):
        assert len(data) == 2
        vertices, triangles = data
        if isinstance(vertices, Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(triangles, Tensor):
            triangles = triangles.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    else:
        mesh = data

    mesh.export(str(file))
    # fout = open(name, 'w')
    # fout.write("ply\n")
    # fout.write("format ascii 1.0\n")
    # fout.write("element vertex " + str(len(vertices)) + "\n")
    # fout.write("property float x\n")
    # fout.write("property float y\n")
    # fout.write("property float z\n")
    # fout.write("element face " + str(len(triangles)) + "\n")
    # fout.write("property list uchar int vertex_index\n")
    # fout.write("end_header\n")
    # for ii in range(len(vertices)):
    #     fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    # for ii in range(len(triangles)):
    #     fout.write("3 " + str(triangles[ii, 0]) + " " + str(triangles[ii, 1]) + " " + str(triangles[ii, 2]) + "\n")
    # fout.close()
    return


def save_obj(filename, vertices, triangles):
    """
    Exports a mesh in the (.obj) format.
    """

    with open(filename, 'w') as fh:
        for v in vertices:
            fh.write("v {} {} {}\n".format(*v))

        for f in triangles:
            f = f + 1
            fh.write("f {} {} {}\n".format(*f))


def save_off(filename, vertices, triangles):
    """
    Exports a mesh in the (.off) format.
    """

    with open(filename, 'w') as fh:
        fh.write('OFF\n')
        fh.write('{} {} 0\n'.format(len(vertices), len(triangles)))

        for v in vertices:
            fh.write("{} {} {}\n".format(*v))

        for f in triangles:
            fh.write("3 {} {} {}\n".format(*f))


def save_glb(filename, *args):
    from my_ext.structures import Mesh
    import open3d as o3d
    from my_ext.utils.open3d_utils import to_open3d_type
    assert len(args) == 1
    mesh = args[0]
    assert isinstance(mesh, Mesh)
    o3d_mesh = o3d.geometry.TriangleMesh(to_open3d_type(mesh.v_pos.double()), to_open3d_type(mesh.f_pos.int()))
    if mesh.v_clr is not None:
        o3d_mesh.vertex_colors = to_open3d_type(mesh.v_clr[..., :3].double())
    o3d.io.write_triangle_mesh(str(filename), o3d_mesh)
