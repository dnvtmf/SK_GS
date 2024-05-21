import numpy as np
import torch
from torch import Tensor

from my_ext.ops_3d.misc import dot, normalize
from my_ext.utils.io.mesh import load_mesh
from my_ext.structures.material_texture import Material, merge_materials, MultiTexture2D


class Mesh:
    attr_names = ['v_pos', 'f_pos', 'v_nrm', 'f_nrm', 'v_tex', 'f_tex', 'v_tng', 'f_tng', 'v_clr', 'f_mat', 'material']

    def __init__(
        self,
        v_pos: Tensor = None,
        f_pos: Tensor = None,
        v_nrm: Tensor = None,
        f_nrm: Tensor = None,
        v_tex: Tensor = None,
        f_tex: Tensor = None,
        v_tng: Tensor = None,
        f_tng: Tensor = None,
        v_clr: Tensor = None,
        f_mat: Tensor = None,
        material: Material = None,
        base: 'Mesh' = None,
    ):
        self.v_pos = v_pos  # value of position
        self.v_nrm = v_nrm  # value of vertices normal
        self.v_tex = v_tex  # value of texture
        self.v_tng = v_tng  # value of tangents
        self.v_clr = v_clr  # value of vertices color
        self.f_pos = f_pos  # faces of vertices
        self.f_nrm = f_nrm  # faces of normal
        self.f_tex = f_tex  # faces of texture
        self.f_tng = f_tng  # faces of tangents
        self.f_mat = f_mat  # indices of texures
        self.material = material

        if base is not None:
            self.copy_none(base)

    @classmethod
    def load(cls, filename, mtl=True, merge_mtl=False) -> 'Mesh':
        data = load_mesh(filename, mtl=mtl, loader='my')
        # from tree_segmentation.extension.utils import show_shape
        # print(show_shape(data))
        kwargs = {'v_pos': data['v_pos'], 'f_pos': data['f_pos']}
        if mtl and 'materials' in data and len(data['materials']) > 0:
            if len(data['materials']) == 1:
                material = Material.from_mtl(data['materials'][0])
            elif merge_mtl:
                mtls = [Material.from_mtl(mtl) for mtl in data['materials']]
                material, v_tex, f_tex = merge_materials(mtls, data['v_tex'], data['f_tex'], data['f_mat'])
                data['v_tex'] = v_tex
                data['f_tex'] = f_tex
            else:
                kwargs['f_mat'] = data['f_mat']
                material = Material.from_mtls(data['materials'])
            kwargs['material'] = material
        if 'v_tex' in data:
            kwargs['v_tex'] = data['v_tex']
            kwargs['f_tex'] = data['f_tex']
        if 'v_nrm' in data:
            kwargs['v_nrm'] = data['v_nrm']
            kwargs['f_nrm'] = data['f_nrm'] if 'f_nrm' in data else data['f_pos']
        if 'v_clr' in data:
            v_clr = data['v_clr']
            if v_clr.dtype == torch.uint8:
                v_clr = v_clr.float() / 255.
            kwargs['v_clr'] = v_clr
        return cls(**kwargs)

    @classmethod
    def from_open3d(cls, mesh) -> 'Mesh':
        data = {
            'v_pos': torch.from_numpy(np.asarray(mesh.vertices)).float(),
            'f_pos': torch.from_numpy(np.asarray(mesh.triangles)).int(),
        }
        if mesh.has_vertex_colors():
            data['v_clr'] = torch.from_numpy(np.asarray(mesh.vertex_colors)).float()
        if mesh.has_vertex_normals():
            data['v_nrm'] = torch.from_numpy(np.asarray(mesh.vertex_normals)).float()
        if mesh.has_textures():
            assert NotImplementedError
        if mesh.has_triangle_uvs():
            assert NotImplementedError
        if mesh.has_triangle_normals():
            assert NotImplementedError
        return cls(**data)

    def to_open3d(self):
        import open3d as o3d
        from my_ext.utils import to_open3d_type
        mesh = o3d.geometry.TriangleMesh(vertices=to_open3d_type(self.v_pos), triangles=to_open3d_type(self.f_pos))
        if self.v_clr is not None:
            mesh.vertex_colors = to_open3d_type(self.v_clr)
        if self.v_nrm is not None:
            mesh.vertex_normals = to_open3d_type(self.v_nrm)
        return mesh

    def to_trimesh(self):
        import trimesh
        return trimesh.Trimesh(
            vertices=self.v_pos.cpu().numpy(),
            faces=self.f_pos.cpu().numpy(),
            # face_normals=self.f_nrm.cpu().numpy() if self.f_nrm is not None else None,
            # vertex_normals=self.v_nrm.cpu().numpy() if self.f_nrm is not None else None,
            vertex_colors=self.v_clr.cpu().numpy() if self.v_clr is not None else None,
        )

    @classmethod
    def from_trimesh(cls, mesh) -> 'Mesh':
        import trimesh.visual
        mesh: trimesh.Trimesh = mesh
        data = {
            'v_pos': torch.from_numpy(mesh.vertices.copy()).float(),
            'f_pos': torch.from_numpy(mesh.faces.copy()).int(),
        }
        data['v_nrm'] = torch.from_numpy(mesh.vertex_normals.copy()).float()
        data['f_nrm'] = data['f_pos'].clone()
        visual = mesh.visual
        if isinstance(visual, trimesh.visual.ColorVisuals):
            data['v_clr'] = torch.from_numpy(visual.vertex_colors.copy()).float() / 255.
        else:
            raise NotImplementedError()
        return cls(**data)

    def check(self):
        assert self.v_pos.ndim == 2 and self.v_pos.shape[1] == 3
        assert self.f_pos.ndim == 2 and self.f_pos.shape[1] == 3
        assert 0 <= self.f_pos.min() and self.f_pos.max() < len(self.v_pos)
        if self.f_nrm is not None:
            assert self.v_nrm.ndim == 2 and self.v_nrm.shape[1] == 3
            assert self.f_nrm.ndim == 2 and self.f_nrm.shape[1] == 3
            assert 0 <= self.f_nrm.min() and self.f_nrm.max() < len(self.v_nrm)
        if self.f_tex is not None:
            assert self.v_tex.ndim == 2 and self.v_tex.shape[1] == 2
            assert self.f_tex.ndim == 2 and self.f_tex.shape[1] == 3
            assert 0 <= self.f_tex.min() and self.f_tex.max() < len(self.v_tex)
        if self.f_tng is not None:
            assert self.v_tng.ndim == 2 and self.v_tng.shape[1] == 3
            assert self.f_tng.ndim == 2 and self.f_tng.shape[1] == 3
            assert 0 <= self.f_tng.min() and self.f_tng.max() < len(self.v_tng)
        if self.material is not None:
            for k in self.material.keys():
                if isinstance(self.material[k], MultiTexture2D):
                    assert self.f_mat is not None
                    assert self.f_mat.ndim == 1 and self.f_mat.shape[0] == self.f_pos.shape[0]
                    assert 0 <= self.f_mat.min() and self.f_mat.max() < len(self.material[k])

    def save(cls, filename):
        raise NotImplementedError()

    def copy_none(self, other):
        for attr in self.attr_names:
            v = getattr(self, attr)
            if v is None:
                setattr(self, attr, getattr(other, attr))

    def clone(self):
        kwargs = {'material': self.material}
        for attr in ['v_pos', 'f_pos', 'v_nrm', 'f_nrm', 'v_tex', 'f_tex', 'v_tng', 'f_tng', 'v_clr', 'f_mat']:
            v = getattr(self, attr)
            if v is not None:
                kwargs[attr] = v.clone().detach()
        return Mesh(**kwargs)

    def to(self, device=None, **kwargs):
        for attr in self.attr_names:
            v = getattr(self, attr)
            if hasattr(v, 'to'):
                setattr(self, attr, v.to(device, **kwargs))
        return self

    def cuda(self, device=None):
        return self.to('cuda' if device is None else device)

    def cpu(self):
        return self.to('cpu')

    def int(self):
        for attr in ['f_pos', 'f_nrm', 'f_tex', 'f_tng', 'f_mat']:
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.int())
        return self

    def long(self):
        for attr in ['f_pos', 'f_nrm', 'f_tex', 'f_tng', 'f_mat']:
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.long())
        return self

    def float(self):
        for attr in ['v_pos', 'v_nrm', 'v_tex', 'v_tng', 'v_clr']:
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.float())
        return self

    def bound(self):
        return torch.stack(self.v_pos.view(-1, 3).aminmax(dim=0), dim=0)

    @property
    def AABB(self):
        return self.bound()

    def compuate_normals_(self, force=False):
        """Simple smooth vertex normal computation"""
        if self.f_nrm is not None and self.f_nrm is not None and not force:  # skip when face normal exists!
            return self
        i0, i1, i2 = self.f_pos.unbind(-1)

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, v_nrm.new_tensor([0.0, 0.0, 1.0]))
        v_nrm = normalize(v_nrm, dim=-1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        self.v_nrm = v_nrm
        self.f_nrm = self.f_pos.clone()
        return self

    def compuate_normals(self, force=False):
        return Mesh(base=self).compuate_normals_(force)

    def compute_tangents_(self, force=False):
        """Compute tangent space from texture map coordinates
        Follows http://www.mikktspace.com/ conventions
        """
        if self.f_tng is not None and self.v_tng is not None and not force:
            return self
        if self.v_tex is None or self.f_tex is None:
            print('No texture vertices and/or faces')
            return self
        pos = [self.v_pos[self.f_pos[:, i]] for i in range(0, 3)]
        tex = [self.v_tex[self.f_tex[:, i]] for i in range(0, 3)]
        vn_idx = [self.f_nrm[:, i] for i in range(0, 3)]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1

        tangents = tangents / tansum.clamp(1)  # tansum == 0 means vertex is not used

        # Normalize and make sure tangent is perpendicular to normal
        tangents = normalize(tangents)
        tangents = normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))
        self.v_tng = tangents
        self.f_tng = self.f_nrm.clone()
        return self

    def compute_tangents(self, force=False):
        return Mesh(base=self).compute_tangents_(force)

    @torch.no_grad()
    def unit_size(self):
        """Align base mesh to reference mesh:move & rescale to match bounding boxes."""
        vmin_max = self.AABB
        scale = 2 / torch.max(vmin_max[1] - vmin_max[0]).item()
        v_pos = self.v_pos - (vmin_max[1] + vmin_max[0]) / 2  # Center mesh on origin
        v_pos = v_pos * scale  # Rescale to unit size
        return Mesh(v_pos, base=self)

    def center_by_reference(self, aabb=None, scale=2.):
        """Center & scale mesh for rendering"""
        if aabb is None:
            aabb = self.AABB
        center = (aabb[0] + aabb[1]) * 0.5
        scale = scale / torch.max(aabb[1] - aabb[0]).item()
        v_pos = (self.v_pos - center[None, ...]) * scale
        return Mesh(v_pos, base=self)

    def center_(self, aabb=None, scale=1.):
        """Center & scale mesh for rendering"""
        if aabb is None:
            aabb = self.AABB
        center = (aabb[0] + aabb[1]) * 0.5
        self.v_pos = (self.v_pos - center[None, ...]) * scale
        return self

    def center(self, aabb=None, scale=1.):
        return Mesh(base=self).center_(aabb, scale)

    def __repr__(self) -> str:
        s = [
            f"vertices={len(self.v_pos)}",
            f"faces={len(self.f_pos)}",
            None if self.v_clr is None else 'clr',
            None if self.v_tex is None else 'tex',
            None if self.v_nrm is None else 'nrm',
            None if self.v_tng is None else 'tng',
            None if self.f_mat is None else 'f_mat',
            None if self.material is None else f"mat={list(self.material.keys())}",
        ]
        return f"{self.__class__.__name__}({', '.join(si for si in s if si is not None)})"

    def merge(self, *meshes: 'Mesh'):
        d = {'material': self.material}  # TODO: merge material
        for attr in ['pos', 'tex', 'nrm', 'tng']:
            v = [getattr(self, f"v_{attr}")]  # vertex
            f = [getattr(self, f"f_{attr}")]  # faces
            if v[0] is None:
                continue
            n = v[0].shape[0]
            for mesh_i in meshes:
                if getattr(mesh_i, f"f_{attr}", None) is not None:
                    f.append(getattr(mesh_i, f"f_{attr}") + n)
                if getattr(mesh_i, f"v_{attr}", None) is not None:
                    v.append(getattr(mesh_i, f"v_{attr}"))
                    n += v[-1].shape[0]
            if len(v) == len(meshes) + 1 and len(f) == len(meshes) + 1:
                d[f"v_{attr}"] = torch.cat(v, dim=0)
                d[f"f_{attr}"] = torch.cat(f, dim=0)
        return Mesh(**d)
