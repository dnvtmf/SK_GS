"""
The Material Library File using in .obj 
材质库文件，描述的是物体的材质信息，ASCII存储

Reference:
    - http://paulbourke.net/dataformats/mtl/
    - https://www.fileformat.info/format/material/
    - https://people.computing.clemson.edu/~dhouse/courses/405/docs/brief-mtl-file-format.html
"""
from typing import Union, List
import re
from pathlib import Path

from PIL import Image
import numpy as np

from .image import save_image

__all__ = ['material_extension', 'MTL', 'load_mtl', 'save_mtl']

material_extension = '.mtl'  # 后缀名


class MTL:
    """The Material Library File"""

    def __init__(self, name='_default_mat') -> None:
        self.name = name
        self.Kd = np.array([1., 1., 1.], dtype=np.float32)  # 漫反射光照（Diffuse Lighting） [0, 1]
        self.Ka = np.array([0., 0., 0.], dtype=np.float32)  # 环境光照（Ambient Lighting）  [0, 1]
        self.Ks = np.array([0., 0., 0.], dtype=np.float32)  # 镜面光照（Specular Lighting） [0, 1]
        self.Ns: float = None  # 镜面高光的反光度参数（Shininess) [0, 1000]
        self.map_Ka = None
        self.map_Kd = None
        self.map_Ks = None
        self.bump = None
        self.illum = 0  # 计算阴影的光照模型 [0-10]
        self.other = {}

    def set_color(self, data, name='Kd'):
        """材质颜色 
        [x] Ks r g b
        [ ] Ks spectral file.rfl factor
        [ ] Ks xyz x y z
        """
        assert name.lower() in ['ka', 'kd', 'ks']
        assert len(data) == 3
        setattr(self, f"K{name[1]}", np.array(list(map(float, data)), dtype=np.float32))

    def set_Ns(self, data):
        assert len(data) == 1
        self.Ns = float(data[0])
        assert 0 <= self.Ns <= 1000.

    def set_illum(self, data):
        assert len(data) == 1
        self.illum = int(data[0])
        assert 0 <= self.illum <= 10

    def set_texture(self, folder: Path, data, name='map_kd'):
        """纹理映射
        仅支持图片文件"""
        name = name.lower()
        assert name in ['map_ka', 'map_kd', 'map_ks', 'bump']
        if name.startswith('map_k'):
            name = f"map_K{name[5:]}"

        assert len(data) == 1
        texture_file = folder.joinpath(data[0])
        filename = texture_file.stem
        ext = texture_file.suffix

        def _load_image(file) -> np.ndarray:
            # image = load_image(file)
            image = np.array(Image.open(file).convert('RGB'))
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.
            # print('MTL load image:', image.shape)
            return image

        if texture_file.with_name(filename + '_0' + ext).exists():
            mips = []
            while texture_file.with_name(filename + f'_{len(mips)}' + ext).exists():
                mips.append(_load_image(texture_file.with_name(filename + f'_{len(mips)}' + ext)))
            texture = mips
        else:
            if not texture_file.is_file():
                raise FileNotFoundError(f'Can not found texture file: {texture_file}')
            texture = _load_image(texture_file)

        setattr(self, name, texture)


def load_mtl(file: Union[str, Path]) -> List[MTL]:
    file = Path(file).expanduser()
    mtl_dir = file.parent

    with file.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    material = None
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = MTL(data[0])
            materials += [material]
        elif materials:
            if prefix in ['map_kd', 'map_ka', 'map_ks', 'bump']:
                material.set_texture(mtl_dir, data, prefix)
            elif prefix in ['ka', 'kd', 'ks']:
                material.set_color(data, prefix)
            elif prefix == 'ns':
                material.set_Ns(data)
            elif prefix == 'illum':
                material.set_illum(data)
            else:
                material.other[prefix] = data
    return materials


_texture_name_map = {
    'map_Ka': 'texture_ka',
    'map_Ks': 'texture_kd',
    'map_Kd': 'texture_ks',
    'bump': 'texture_normal',
}


def save_mtl(file: Union[str, Path], *mtls: Union[MTL, dict], texture_dir=None):
    file = Path(file).expanduser().with_suffix(material_extension)
    texture_dir = file.parent if texture_dir is None else Path(texture_dir)
    if len(mtls) == 0:
        mtls = [MTL()]
    with file.open("w") as f:
        for mtl in mtls:
            name = getattr(mtl, 'name', 'defaultMat')
            f.write(f'newmtl {name}\n')
            for attr in ['Ka', 'Kd', 'Ks']:
                value = getattr(mtl, attr, None)
                if value is not None:
                    f.write(f"{attr} {value[0]} {value[1]} {value[2]}\n")
            for attr in ['Ns', 'illum']:
                value = getattr(mtl, attr, None)
                if value is not None:
                    f.write(f"{attr} {value}\n")
            for attr in ['map_Ka', 'map_Kd', 'map_Ks', 'bump']:
                value = getattr(mtl, attr, None)
                if value is not None:
                    # TODO:
                    assert not isinstance(value, (tuple, list)), f"Not support mip texture Now"
                    filename = _texture_name_map[attr] + ('' if len(mtls) == 1 else '_' + name)
                    filename = texture_dir.joinpath(filename + '.png')
                    save_image(filename, value)
                    f.write(f"{attr} {filename.relative_to(file.parent)}\n")
