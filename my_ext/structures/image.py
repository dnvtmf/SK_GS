import math
import numbers
from typing import Tuple, Union

from PIL import ImageOps
import PIL.Image as Img
import PIL.ImageFile
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from my_ext.structures import Structure2D, Restoration
from my_ext.utils import n_tuple

__all__ = ['Image']
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class Image(Structure2D):
    CHANNELS = 3
    PAD_VALUE = 0

    def __init__(self, img: Union[np.ndarray, Img.Image], backend='cv2', infos=None):
        super(Image, self).__init__(infos)
        assert backend in ['cv2', 'PIL']
        self.img = img
        if isinstance(img, np.ndarray):
            self.backend = 'cv2'
            assert self.img.ndim == 3 and self.img.shape[2] == Image.CHANNELS
        elif isinstance(img, Img.Image):
            self.backend = 'PIL'
            assert len(self.img.mode) == Image.CHANNELS
        else:
            raise NotImplementedError(f'type {type(img)} is not numpy or Image')

        self.img: Union[np.ndarray, Img.Image]
        if self.backend != backend:
            self.convert(backend)

    @staticmethod
    def load(filename, backend='cv2'):
        with open(filename, 'rb') as f:
            img = Img.open(f)
            if Image.CHANNELS == 1:
                img = img.convert('L')
            elif Image.CHANNELS == 3:
                img = img.convert('RGB')
            elif Image.CHANNELS == 4:
                img = img.convert('RGBA')
        if backend == 'cv2':
            img = np.array(img)
        return Image(img, backend=backend)

    @classmethod
    def load_by_cv2(cls, filename, backend='cv2'):
        if Image.CHANNELS == 1:
            img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        elif Image.CHANNELS == 3:
            img = cv2.imread(str(filename), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
            assert img.shape[-1] == Image.CHANNELS
        return cls(Img.fromarray(img) if backend == 'PIL' else img, backend=backend)

    def clone(self):
        return Image(self.img, self.backend, self.infos)

    def convert_(self, mode):
        if self.backend != mode:
            if mode == 'cv2':
                self.img = np.array(self.img)
            else:
                self.img = Img.fromarray(self.img)
            self.backend = mode
        return self

    def convert(self, mode):
        return Image(self.img, self.backend).convert_(mode)

    @property
    def size(self) -> Tuple[int, int]:
        if self.backend == 'cv2':
            h, w, = self.img.shape[:2]
        else:
            w, h = self.img.size
        return w, h

    @property
    def channels(self):
        return self.img.shape[2] if self.backend == 'cv2' else len(self.img.mode)

    def get_interpolation(self, interpolation="linear", backend=None):
        if backend is None:
            backend = self.backend
        interpolation = interpolation.lower()
        if backend == 'PIL':
            return {
                "nearest": Img.NEAREST,
                "bilinear": Img.BILINEAR,
                "linear": Img.LINEAR,
                "cubic": Img.CUBIC,
                "bicubic": Img.BICUBIC
            }[interpolation]
        else:
            return {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'bilinear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'bicubic': cv2.INTER_CUBIC,
            }[interpolation]

    @staticmethod
    def get_cv2_pad_mode(mode='constant'):
        return {'constant': cv2.BORDER_CONSTANT,
                'edge': cv2.BORDER_REPLICATE,
                'reflect': cv2.BORDER_REFLECT_101,
                'symmetric': cv2.BORDER_REFLECT
                }[mode]

    def resize_(self, size, interpolation="linear", *args, **kwargs):
        if self.backend == 'cv2':
            self.img = cv2.resize(self.img, size, interpolation=self.get_interpolation(interpolation))
        else:
            self.img = self.img.resize(size, self.get_interpolation(interpolation))
        return self

    def resize(self, size, interpolation="linear", *args, **kwargs):
        return Image(self.img, self.backend, self.infos).resize_(size, interpolation, *args, **kwargs)

    def crop_(self, x, y, w, h, *args, **kwargs):
        if self.backend == 'cv2':
            self.img = self.img[y:y + h, x:x + w, :]
        else:
            self.img = self.img.crop((x, y, x + w, y + h))
        return self

    def crop(self, x, y, w, h, *args, **kwargs):
        return Image(self.img, self.backend, self.infos).crop_(x, y, w, h, *args, **kwargs)

    def pad_(self, padding, img_fill=None, img_padding_mode='constant', *args, **kwargs):
        if img_fill is None:
            img_fill = self.PAD_VALUE
        padding = self._get_padding_param(padding)
        if self.backend == 'cv2':
            # h, w, d = self.img.shape
            # nw = w + padding[0] + padding[2]
            # nh = h + padding[1] + padding[3]
            # new_img = np.ones((nh, nw, d), self.img.dtype) * fill
            # new_img[padding[1]:padding[1] + h, padding[0]:padding[0] + w, :] = self.img[:, :, :]
            new_img = cv2.copyMakeBorder(self.img, padding[1], padding[3], padding[0], padding[2],
                self.get_cv2_pad_mode(img_padding_mode), value=img_fill)
            self.img = new_img
        else:
            self.img = F.pad(self.img, padding, img_fill, img_padding_mode)
        return self

    def pad(self, padding, img_fill=None, img_padding_mode='constant', *args, **kwargs):
        return Image(self.img, self.backend, self.infos).pad_(padding, img_fill, img_padding_mode, *args, **kwargs)

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        img = self.img
        if self.backend == 'cv2':
            if horizontal:
                img = np.flip(img, axis=1)
            if vertical:
                img = np.flip(img, axis=0)
        else:
            if horizontal:
                img = img.transpose(Img.FLIP_LEFT_RIGHT)
            if vertical:
                img = img.transpose(Img.FLIP_TOP_BOTTOM)
        self.img = img
        return self

    def flip(self, horizontal=False, vertical=False, *args, **kwargs):
        return self.clone().flip_(horizontal, vertical, *args, **kwargs)

    def adjust_brightness(self, brightness_factor):
        """Adjust brightness of an Image.
        Args:
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.
        """
        img = self.img
        if self.backend == 'cv2':
            table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
            # same thing but a bit slower
            # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
            if img.shape[2] == 1:
                img = cv2.LUT(img, table)[:, :, np.newaxis]
            else:
                img = cv2.LUT(img, table)
        else:
            img = F.adjust_brightness(img, brightness_factor)
        return Image(img, self.backend, self.infos)

    def adjust_contrast(self, contrast_factor):
        """Adjust contrast of an image.
        Args:
            contrast_factor (float): How much to adjust the contrast. Can be any
                non negative number. 0 gives a solid gray image, 1 gives the
                original image while 2 increases the contrast by a factor of 2.
        Returns:
            numpy ndarray: Contrast adjusted image.
        """
        # much faster to use the LUT construction than anything else I've tried
        # it's because you have to change dtypes multiple times
        img = self.img
        if self.backend == 'cv2':
            table = np.array([(i - 74) * contrast_factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')
            # enhancer = ImageEnhance.Contrast(img)
            # img = enhancer.enhance(contrast_factor)
            if img.shape[2] == 1:
                img = cv2.LUT(img, table)[:, :, np.newaxis]
            else:
                img = cv2.LUT(img, table)
        else:
            img = F.adjust_contrast(self.img, contrast_factor)
        return Image(img, self.backend, self.infos)

    def adjust_saturation(self, saturation_factor):
        """Adjust color saturation of an image.
        Args:
            saturation_factor (float):  How much to adjust the saturation. 0 will
                give a black and white image, 1 will give the original image while
                2 will enhance the saturation by a factor of 2.
        Returns:
            numpy ndarray: Saturation adjusted image.
        """
        # cv2 is ~10ms slower than PIL!
        img = Img.fromarray(self.img) if self.backend == 'cv2' else self.img
        img = F.adjust_saturation(img, saturation_factor)
        return Image(img, self.backend, self.infos)

    def adjust_hue(self, hue_factor):
        """Adjust hue of an image.
        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.
        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.
        See `Hue`_ for more details.
        .. _Hue: https://en.wikipedia.org/wiki/Hue
        Args:
            hue_factor (float):  How much to shift the hue channel. Should be in
                [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                HSV space in positive and negative direction respectively.
                0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                with complementary colors while 0 gives the original image.
        Returns:
            numpy ndarray: Hue adjusted image.
        """
        # After testing, found that OpenCV calculates the Hue in a call to
        # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

        # This function takes 160ms! should be avoided
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

        img = Img.fromarray(self.img) if self.backend == 'cv2' else self.img
        if self.backend == 'cv2':
            input_mode = img.mode
            assert input_mode not in {'L', '1', 'I', 'F'}

            h, s, v = img.convert('HSV').split()

            np_h = np.array(h, dtype=np.uint8)
            # uint8 addition take cares of rotation across boundaries
            with np.errstate(over='ignore'):
                np_h += np.uint8(hue_factor * 255)
            h = Img.fromarray(np_h, 'L')
            img = Img.merge('HSV', (h, s, v)).convert(input_mode)
        else:
            img = F.adjust_hue(self.img, hue_factor)
        return Image(img, self.backend, self.infos)

    def adjust_gamma(self, gamma, gain=1):
        r"""Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        .. math::
            I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
        See `Gamma Correction`_ for more details.
        .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
        Args:
            gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
                gamma larger than 1 make the shadows darker,
                while gamma smaller than 1 make dark regions lighter.
            gain (float): The constant multiplier.
        """
        if self.backend == 'cv2':
            img = self.img
            if gamma < 0:
                raise ValueError('Gamma should be a non-negative real number')
            # from here
            # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
            table = np.array([((i / 255.0) ** gamma) * 255 * gain for i in np.arange(0, 256)]).astype('uint8')
            if img.shape[2] == 1:
                img = cv2.LUT(img, table)[:, :, np.newaxis]
            else:
                img = cv2.LUT(img, table)
        else:
            img = F.adjust_gamma(self.img, gamma, gain)
        return Image(img, self.backend, self.infos)

    def color_jitter(self, brightness=1., contrast=1., saturation=1.0, hue=0., value=1.0, first_do_contrast=True):
        """

        :param brightness: How much to adjust the brightness.
            Can be any non negative number.
            0 gives a black image,
            1 gives the original image while
            2 increases the brightness by a factor of 2.
        :param contrast: How much to adjust the contrast. Can be any non negative number.
            0 gives a solid gray image,
            1 gives the original image while
            2 increases the contrast by a factor of 2.
        :param saturation: How much to adjust the saturation.
            0 will give a black and white image,
            1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        :param hue: [-0.5, 0.5]
        :param value: [0, 2]
        :param first_do_contrast:
        :return:
        """
        if Image.CHANNELS == 1:
            raise NotImplementedError
        img = self.img
        if self.backend == 'cv2':
            c4 = None
            if Image.CHANNELS == 4:
                c4 = img[:, :, 3:4]
                img = img[:, :, :3]
            if first_do_contrast:
                table = np.arange(256, dtype=np.float).repeat(3).reshape(1, -1, 3)  # type: np.ndarray
                table = (table * contrast + 128. * (brightness - 1.))
                table = np.clip(table, 0, 255).astype(np.uint8)
                img = cv2.LUT(img, table)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            table = np.arange(256, dtype=np.float).repeat(3).reshape(1, -1, 3)  # type: np.ndarray
            hue = int(180. * hue) if hue >= 0 else 180. * (1. + hue)
            table[:, :, 0] = (table[:, :, 0] + hue) % 180
            table[:, :, 1] = table[:, :, 1] * saturation
            table[:, :, 2] = table[:, :, 2] * value
            table = np.clip(table, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, table)
            # img[:, :, 0] = (img[:, :, 0] + int(180. * hue)) % 180
            # img[:, :, 1] = (img[:, :, 1] * saturation).clip(0, 255)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            if not first_do_contrast:
                table = np.arange(256, dtype=np.float).repeat(3).reshape(1, -1, 3)  # type: np.ndarray
                table = (table * contrast + 128. * (brightness - 1.))
                table = np.clip(table, 0, 255).astype(np.uint8)
                img = cv2.LUT(img, table)
            if Image.CHANNELS == 4:
                img = np.concatenate([img, c4], axis=-1)
        else:
            assert Image.CHANNELS == 3
            if first_do_contrast:
                img = F.adjust_contrast(img, contrast)
                img = F.adjust_brightness(img, brightness)
            img = F.adjust_hue(img, hue)
            img = F.adjust_saturation(img, saturation)
            if not first_do_contrast:
                img = F.adjust_contrast(img, contrast)
                img = F.adjust_brightness(img, brightness)
        return Image(img, self.backend, self.infos)

    def auto_contrast(self, cutoff: float = 0, ignore: int = None):
        """
        最大化（标准化）图像对比度。此函数计算输入图像的直方图，删除 cutoff 柱状图中最亮和最暗像素的百分比，并重新映射图像，
        使最暗的像素变为黑色（0），最亮的像素变为白色（255）。
        :param cutoff: float or tuple(float), 从柱状图中截取多少百分比。
        :param ignore:背景像素值（无背景使用None）。
        :return:
        """
        if self.backend == 'PIL':
            return Image(ImageOps.autocontrast(self.img, cutoff, ignore), 'PIL', self.infos)
        else:
            lut = np.arange(256, dtype=np.float).repeat(self.channels).reshape(1, -1, self.channels)
            for c in range(self.img.shape[2]):
                h = np.bincount(self.img[:, :, c].reshape(-1), minlength=256)  # histogram
                if ignore is not None:
                    h[ignore] = 0
                if cutoff:
                    cutoff = n_tuple(cutoff, 2)
                    n_sum = h.sum()
                    # cut off low ends
                    cut = int(cutoff[0] * n_sum // 100.)
                    for i in range(256):
                        if h[i] < cut:
                            cut -= h[i]
                            h[i] = 0
                        else:
                            h[i] -= cut
                            break
                    # cut off high ends
                    cut = int(cutoff[1] * n_sum // 100.)
                    for i in range(255, -1, -1):
                        if h[i] < cut:
                            cut -= h[i]
                            h[i] = 0
                        else:
                            h[i] -= cut
                            break
                non_zero, = h.nonzero()
                if len(non_zero) > 1:
                    lo = non_zero.min()
                    hi = non_zero.max()
                    lut[0, :, c] = (lut[0, :, c] - lo) / (hi - lo) * 255
            lut = np.clip(lut, 0, 255).astype(np.uint8)
            lut = np.ascontiguousarray(lut)
            img = cv2.LUT(self.img, lut)
            return Image(img, 'cv2', self.infos)

    def invert(self):
        if self.backend == 'PIL':
            return Image(ImageOps.invert(self.img), 'PIL', self.infos)
        else:
            lut = 255 - np.arange(256, dtype=np.uint8)
            return Image(cv2.LUT(self.img, lut), 'cv2', self.infos)

    def equalize(self):
        """
        均衡图像直方图。此函数将非线性映射应用于输入图像，以便在输出图像中创建统一的灰度值分布。
        :return:
        """
        if self.backend == 'PIL':
            return Image(ImageOps.equalize(self.img), 'PIL', self.infos)
        else:
            nc = self.channels
            lut = np.arange(256, dtype=np.int32).repeat(nc).reshape(1, -1, nc)
            for c in range(nc):
                h = np.bincount(self.img[:, :, c].reshape(-1), minlength=256)  # histogram
                idx = h.nonzero()[0]
                if len(idx) <= 1:
                    continue
                step = (h.sum() - h[idx[-1]]) // 255
                if step:
                    n = step // 2
                    lut[0, :, c] = (h.cumsum() - h + n) // step
            lut = lut.astype(np.uint8)
            return Image(cv2.LUT(self.img, lut), 'cv2', self.infos)

    def solarize(self, threshold=128):
        """反转高于阈值的所有像素值。"""
        if self.backend == 'PIL':
            return Image(ImageOps.solarize(self.img), 'PIL', self.infos)
        else:
            lut = np.arange(256, dtype=np.uint8)
            lut = np.where(lut < threshold, lut, 255 - lut)
            return Image(cv2.LUT(self.img, lut), 'cv2', self.infos)

    def posterize(self, bits: int):
        """减少每个颜色通道的位数。"""
        if self.backend == 'PIL':
            return Image(ImageOps.posterize(self.img, bits), 'PIL', self.infos)
        else:
            mask = ~(2 ** (8 - bits) - 1)
            return Image((self.img & mask).astype(np.uint8), 'cv2', self.infos)

    def get_PIL_affine_matrix(self, angle=0., translate=(0., 0.), scale=1., shear=(0., 0.), center=None):
        """Apply affine transformation on the image keeping image center invariant

            Code is copied and modified from torchvision.transforms.functional

        Args:
          angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
          translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
          scale (float): overall scale
          shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
          If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
          the second value corresponds to a shear parallel to the y axis.
          center (tuple or list): ...
        """
        #
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, s, (sx, sy)) =
        #       = R(a) * S(s) * SHy(sy) * SHx(sx)
        #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
        #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
        #         [ 0                    , 0                                      , 1 ]
        #
        # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
        # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
        #          [0, 1      ]              [-tan(s), 1]
        #
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "Argument translate should be a list or tuple of length 2"

        assert scale > 0.0, "Argument scale should be positive"
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing two values. Got {}".format(shear))

        rot = math.radians(-angle)
        sx, sy = [math.radians(-s) for s in shear]

        # cx, cy = (self.size[0] * 0.5 + 0.5, self.size[1] * 0.5 + 0.5)
        cx, cy = (self.size[0] * 0.5, self.size[1] * 0.5) if center is None else center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    def affine(self, affine_matrix, output_size, interpolation="linear", img_fill=None, **kwargs):
        """Apply affine transformation on the image keeping image center invariant

            img_fill (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
        """
        img_fill = self.PAD_VALUE if img_fill is None else img_fill
        img = self.img if self.backend == 'cv2' else np.array(self.img)
        img = cv2.warpAffine(img, affine_matrix[:2], output_size, flags=self.get_interpolation(interpolation, 'cv2'),
            borderValue=img_fill)

        ## PIL first the rotate direction is different from cv2; then PIL need the inverse matrix of affine_matrix
        # from PIL import __version__ as PILLOW_VERSION
        # kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
        # resample = self.get_interpolation(interpolation)
        # affine_matrix = self.get_PIL_affine_matrix()
        # img = self.img.transform(output_size, Img.AFFINE, affine_matrix, resample, **kwargs)
        return Image(img, self.backend, self.infos)

    def get_tensor(
        self, mean=None, std=None, rgb=True, divided255=True, device=None,
        dtype=torch.float32
    ) -> torch.Tensor:
        img = self.img if self.backend == 'cv2' else np.array(self.img)
        if divided255:
            img = img / 255.
        if not rgb:
            img = img[:, :, ::-1]
        if mean is None:
            mean = 0.5
        if std is None:
            std = 0.5
        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).to(device=device, dtype=dtype)

    def restore(self, rt: Restoration):
        result = self
        for t in rt.get_transforms():
            result = getattr(result, t[0])(*t[1:-1], **t[-1])
        return result

    def __repr__(self):
        w, h = self.size
        c = self.channels
        return f"Image[{w}x{h}x{c}{self.infos if self.infos else ''}]"

    def draw(self, std=None, mean=None, *args, **kwargs) -> np.ndarray:
        return self.get_image(self.img, std, mean, *args, **kwargs)

    @staticmethod
    def get_image(image, std=None, mean=None, *args, **kwargs) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if std is not None:
                image = image * torch.tensor(std).view(3, 1, 1)
            if mean is not None:
                image = image + torch.tensor(mean).view(3, 1, 1)
            # if image.max() <= 1.:
            image = image * 255
            image = image.byte().numpy()
            image = np.transpose(image, (1, 2, 0))
        elif isinstance(image, Image):
            image.convert('cv2')
            image = image.img
        image = np.array(image)  # ensure image is np.array
        assert image.dtype == np.uint8
        return image


def _test_speed_color_jitter():
    from my_ext.utils import TimeWatcher
    import random
    filename = '/home/wan/Pictures/cat.jpeg'
    f1 = Image(filename, backend='cv2')
    f2 = Image(filename, backend='PIL')
    timer = TimeWatcher()
    for _ in range(100):
        saturation_factor = random.uniform(0, 2)
        brightness_factor = random.uniform(0, 2)
        contrast_factor = random.uniform(0, 2)
        hue_factor = random.uniform(-0.5, 0.5)
        cv_first = random.randint(0, 1)
        # print(f"brightness={brightness_factor:.2f}, contrast={contrast_factor:.2f}, hue={hue_factor:.2f}, "
        #       f"saturation={saturation_factor:.2f}, first_do_contrast={cv_first}")
        timer.start()
        f1.color_jitter(brightness=brightness_factor, contrast=contrast_factor, hue=hue_factor,
            saturation=saturation_factor, first_do_contrast=cv_first)
        timer.log('cv2')
        f2.color_jitter(brightness=brightness_factor, contrast=contrast_factor, hue=hue_factor,
            saturation=saturation_factor, first_do_contrast=cv_first)
        timer.log('pil')
        print('\r%d' % _, end=' ')
    print()
    print(timer)


def _main():
    import matplotlib.pyplot as plt
    filename = '/home/wan/Pictures/cat.jpeg'
    f1 = Image(filename, backend='cv2')
    f2 = Image(filename, backend='PIL')

    def deal_color_jitter():
        brightness_factor = 1
        contrast_factor = 1
        saturation_factor = 1
        hue_factor = 0
        a = f1.color_jitter(brightness=brightness_factor, contrast=brightness_factor, hue=hue_factor,
            saturation=saturation_factor)
        plt.imshow(a.img)
        plt.show()
        b = f2.color_jitter(brightness=brightness_factor, contrast=contrast_factor, hue=hue_factor,
            saturation=saturation_factor)
        plt.imshow(b.img)
        plt.show()
        x = a.img.astype(np.float)
        y = np.array(b.img).astype(np.float)
        print('mean diff=', np.mean(np.abs(x - y)))
        print('max diff=', np.max(np.abs(x - y)))

    def deal_affine(angle=45., translate=(100, 200), scale=(2.0, 2.0), shear=(30, 0.)):
        print(f1.size, f2.size)
        M1 = f1.get_affine_matrix(angle, translate, scale, shear)
        M2 = f2.get_affine_matrix(angle, translate, scale, shear)
        output_size = (500, 500)
        x = f1.affine(M1, output_size, interpolation='bicubic', img_fill=(255, 255, 255))
        y = f2.affine(M2, output_size, interpolation='bicubic', img_fill=(255, 255, 255))
        print(x.size, y.size)
        plt.imshow(x.img)
        plt.show()
        plt.imshow(y.img)
        plt.show()

    def deal(func=None, *args, **kwargs):
        a, b = f1, f2
        if func is not None:
            a = getattr(a, func)(*args, **kwargs)
            b = getattr(b, func)(*args, **kwargs)
        plt.imshow(a.img)
        plt.show()
        plt.imshow(b.img)
        plt.show()

    # deal_color_jitter()
    deal_affine()
    # deal('resize', (400, 200))
    # deal('crop', 50, 50, 200, 100)
    # deal('pad', (10, 20, 30, 40))
    # deal('flip', True)
    # deal('adjust_hue', 0.5)
    # deal('adjust_gamma', 2)
    # deal('adjust_saturation', 2)
    # deal('adjust_brightness', 2)
    # deal('adjust_contrast', 2)
    # print(f1.get_tensor().max())
    # print(f1.get_tensor().min())


def _test_multi_color():
    import matplotlib.pyplot as plt
    RGBA_filename = '/home/wan/Pictures/test.jpg'
    RGB_filename = '/home/wan/Pictures/pic.jpeg'

    def read_and_show(filename):
        img_a = Image(filename, backend='cv2')
        img_b = Image(filename, backend='PIL')
        plt.subplot(2, 1, 1)
        plt.imshow(img_a.img)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(img_b.img)
        plt.axis('off')
        plt.show()
        print(img_a, img_b)

    read_and_show(RGB_filename)
    read_and_show(RGBA_filename)
    Image.CHANNELS = 4
    read_and_show(RGB_filename)
    read_and_show(RGBA_filename)
    Image.CHANNELS = 1
    read_and_show(RGB_filename)
    read_and_show(RGBA_filename)


if __name__ == '__main__':
    # _main()
    # _test_speed_color_jitter()
    _test_multi_color()
