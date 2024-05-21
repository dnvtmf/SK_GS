import torch
from torch import Tensor, nn
import numpy as np


class MarchingTetrahedrons(nn.Module):
    """ Marching tetrahedrons implementation (differentiable)

    adapted from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
    """

    def __init__(self):
        super().__init__()
        triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
            [4, 0, 3, -1, -1, -1],
            [1, 4, 2, 1, 3, 4],
            [3, 1, 5, -1, -1, -1],
            [2, 3, 0, 2, 5, 3],
            [1, 4, 0, 1, 5, 4],
            [4, 2, 5, -1, -1, -1],
            [4, 5, 2, -1, -1, -1],
            [4, 1, 0, 4, 5, 1],
            [3, 2, 0, 3, 5, 2],
            [1, 3, 5, -1, -1, -1],
            [4, 1, 2, 4, 3, 1],
            [3, 0, 4, -1, -1, -1],
            [2, 0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=torch.long) # yapf: disable


        num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
        base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)

        self.register_buffer('triangle_table', triangle_table, persistent=False)
        self.register_buffer('num_triangles_table', num_triangles_table, persistent=False)
        self.register_buffer('base_tet_edges', base_tet_edges, persistent=False)

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx + 1) // 2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device=face_gidx.device),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device=face_gidx.device),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([tex_x, tex_y, tex_x + pad, tex_y, tex_x + pad, tex_y + pad, tex_x, tex_y + pad],
                          dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2), dim=-1).view(-1, 3)

        return uvs, uv_idx

    def forward(self, pos_nx3: Tensor, sdf_n: Tensor, tet_fx4: Tensor):
        r""" Marching tets implementation
        
        Convert discrete signed distance fields encoded on tetrahedral grids to triangle meshes using marching
        tetrahedra algorithm as described in `An efficient method of triangulating equi-valued surfaces by using 
        tetrahedral cells`_. 
        The output surface is differentiable with respect to input vertex positions and the SDF values. 
        For more details and example usage in learning, see `Deep Marching Tetrahedra\: a Hybrid Representation 
        for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.
        
        .. _An efficient method of triangulating equi-valued surfaces by using tetrahedral cells:
            https://search.ieice.org/bin/summary.php?id=e74-d_1_214
        .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
                https://arxiv.org/abs/2111.04276
        """
        device = pos_nx3.device
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device=device)[valid_tets]
        face_gidx = torch.cat(
            (
                tet_gidx[num_triangles == 1] * 2,
                torch.stack((tet_gidx[num_triangles == 2] * 2, tet_gidx[num_triangles == 2] * 2 + 1), dim=-1).view(-1)
            ),
            dim=0,
        )

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets * 2)

        return verts, faces, uvs, uv_idx


def create_unit_tets(n=64):
    """基于分辨率n创造一个四面体网格"""
    vertices = []
    for i in range(-1, n + 2):
        yz = []
        if i % 2 == 1:
            for k in range(n // 4):
                ny = n // 4
                yz.append((np.arange(ny) * 4 + 2, np.full(ny, (k * 4 + 2))))

        else:
            for j in range(n // 2 + 1):
                if j % 2 == 0:
                    not_edge = j != 0 and j != (n // 2)
                    nz = n // 2 + not_edge * 2
                    y, z = np.full(nz, (j * 2)), (np.arange(nz) * 2 + 1 - 2 * not_edge)
                else:
                    y, z = np.full(n // 4 + 1, j * 2), (np.arange(n // 4 + 1) * 4)
                yz.append((y, z) if i // 2 % 2 == 1 else (z, y))

        vertices.append(np.stack([np.full_like(y, i), y, z], axis=-1))
