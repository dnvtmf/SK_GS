# 光照模型
# 约定 l: 光照方向Light direction; n: 法线方向Surface normal; v: 视角方向Viewer direction; t: 切线方向; h: 半程向量(Bisector)
# materials: (ka, kd, ks)
from typing import Union, List, Sequence

import torch
from torch import Tensor

import my_ext.ops_3d.xfm
from .misc import dot, normalize, reflect

__all__ = ['Lambert', 'HalfLambert', 'Phong', 'Blinn_Phong']


def Lambert(n: Tensor, l: Union[Tensor, List[Tensor]], v: Tensor, materials: Sequence[Tensor], **kwargs):
    """
    兰伯特反射(Lambert)是最常见的一种漫反射，它与视角无关，即从不同的方向观察并不会改变渲染结果。
    """
    ka, kd, ks = materials
    l, la, ld, ls = (l, 1, 1, 1) if isinstance(l, Tensor) else l

    diffuse = dot(n, l).clamp(0., 1.)
    return ka * la + diffuse * kd * ld


def HalfLambert(n: Tensor, l: Union[Tensor, List[Tensor]], v: Tensor, materials: Sequence[Tensor], p=8., **kwargs):
    """半兰伯特模型是一个非基于物理的光照模型.
    这个光照模型可以防止像兰伯特模型一样出现“死黑”的部分而导致丢失模型细节，但会使得模型看起来很平。"""
    ka, kd, ks = materials
    l, la, ld, ls = (l, 1, 1, 1) if isinstance(l, Tensor) else l
    # n = normalize(n)
    # l = normalize(l)
    diffuse = torch.pow(dot(n, l) * 0.5 + 0.5, p) * kd
    return ka * la + diffuse * kd * ld


def Phong(n: Tensor, l: Union[Tensor, List[Tensor]], v: Tensor, materials: Sequence[Tensor], shininess=8., **kwargs):
    """Phong光照模型是用来创建高光效果的。

    它认为从物体表面反射出的光线中包括粗糙表面的漫反射与光滑表面高光两部分。"""
    ka, kd, ks = materials
    l, la, ld, ls = (l, 1, 1, 1) if isinstance(l, Tensor) else l
    n = normalize(n)
    v = normalize(v)
    rl = reflect(-l, n)

    diffuse = dot(l, n).clamp_min_(0)  # .pow(shininess)
    speucalr = dot(rl, v).clamp_min_(0).pow(shininess)

    return ka * la + diffuse * kd * ld + speucalr * ks * ls


def Blinn_Phong(
    n: Tensor,
    l: Union[Tensor, List[Tensor]],
    v: Tensor,
    materials: Sequence[Tensor],
    shininess=5.,
    **kwargs
):
    """Blinn-Phong光照模型 对Phong模型的一种优化 高光结果比Phong更"光滑"""
    ka, kd, ks = materials
    l, la, ld, ls = (l, 1, 1, 1) if isinstance(l, Tensor) else l
    n = normalize(n)
    v = normalize(v)
    h = normalize(l + v)

    diffuse = dot(l, n).clamp_min_(0)  # .pow(shininess)
    speucalr = dot(n, h).clamp_min_(0).pow(shininess)
    return ka * la + diffuse * kd * ld + speucalr * ks * ls


@torch.no_grad()
def test():
    from pathlib import Path
    from my_ext import ops_3d
    import nvdiffrast.torch as dr
    import my_ext as ext
    from .normal import compute_shading_normal

    ext.utils.set_printoptions()
    mesh_path = Path('~/data/meshes/lego/lego.obj').expanduser()
    # mesh_path = Path('~/data/meshes/lego/parts.obj').expanduser()
    # mesh_path = Path('~/data/meshes/lego/parts/part_1_00.obj').expanduser()
    # mesh_path = Path('~/data/meshes/spot/spot.obj').expanduser()
    mesh = ext.Mesh.load(mesh_path, mtl=True)  # type: ext.Mesh
    # return
    mesh = mesh.unit_size()
    mesh.cuda()
    mesh.compuate_normals_(False)
    mesh.compute_tangents_()
    mesh.int()
    print(mesh)
    # print('kd:', utils.show_shape(mesh.material['kd'].data))
    # print('kd device:', mesh.material['kd'].data.device)

    glctx = dr.RasterizeCudaContext()

    # light = normalize(torch.randn(3)).cuda()
    # light = normalize(torch.tensor([0, 1., 0])).cuda()
    light = ops_3d.PointLight(location=(0, 2., 0.)).cuda()

    def shading(Tw2v, fovy, size):
        Tw2v = Tw2v.cuda()
        Tv2c = ops_3d.perspective(size=size, fovy=fovy).cuda()
        Tw2c = Tv2c @ Tw2v
        v_dir = normalize(torch.inverse(Tw2v)[..., :3, 3])
        pos = my_ext.ops_3d.xfm.xfm(mesh.v_pos, Tw2c)[None]
        tri = mesh.f_pos.int()
        rast, _ = dr.rasterize(glctx, pos, tri, [512, 512])

        if mesh.f_mat is not None:
            f_mat = mesh.f_mat[rast[..., -1].int() - 1].long()
        else:
            f_mat = None
        points, _ = dr.interpolate(mesh.v_pos[None], rast, mesh.f_pos.int())
        uv, uv_da = dr.interpolate(mesh.v_tex[None], rast, mesh.f_tex.int())
        ka = mesh.material['ka'].sample(uv, f_mat=f_mat)[..., :3].contiguous() if 'ka' in mesh.material else 0
        kd = mesh.material['kd'].sample(uv, f_mat=f_mat)[..., :3].contiguous() if 'kd' in mesh.material else 0
        ks = mesh.material['ks'].sample(uv, f_mat=f_mat)[..., :3].contiguous() if 'ks' in mesh.material else 0

        # nrm, _ = dr.interpolate(mesh.v_nrm[None], rast, mesh.f_nrm.int())
        # nrm = ops_3d.xfm(nrm, torch.inverse(Tw2c).permute(-1, -2))
        # nrm2 = normalize(nrm[..., :3])

        # nrm = ops_3d.xfm(mesh.v_nrm, torch.inverse(Tw2c).permute(-1, -2))
        # nrm, _ = dr.interpolate(nrm[None], rast, mesh.f_nrm.int())
        # nrm1 = normalize(nrm[..., :3])

        nrm = compute_shading_normal(mesh, torch.inverse(Tw2v)[..., :3, 3], rast, None)
        # assert nrm.shape == nrm1.shape and nrm.shape == nrm2.shape
        # print(
        #     'campare normal',
        #     cosine_similarity(nrm1, nrm, dim=-1).abs().mean().item(),
        #     cosine_similarity(nrm2, nrm, dim=-1).abs().mean().item(),
        #     cosine_similarity(nrm, nrm, dim=-1).abs().mean().item(),
        # )

        # img = HalfLambert(nrm, light, kd)
        # img = Phong(nrm, light, v_dir, kd, ks)
        img = Blinn_Phong(nrm, light(points), v_dir, (ka, kd, ks))
        img = torch.where(rast[..., -1:] > 0, img, torch.ones_like(img))
        img = dr.antialias(img, rast, pos, tri)
        return img

    # print(utils.show_shape(img, nrm, light, rast, mesh.v_nrm))
    # plt.imshow(utils.as_np_image(img[0]))
    # plt.show()
    ext.utils.gui.simple_3d_viewer(shading)
