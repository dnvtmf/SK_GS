import math
from copy import deepcopy
from typing import List, Union, Optional, Sequence

import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from my_ext import ops
from my_ext.structures import Structure2D, Restoration
from my_ext.utils import get_colors, n_tuple
from my_ext.distributed import get_world_size, gather_tensor

__all__ = ['BoxList', "gather_bounding_boxes"]


class BoxList(Structure2D):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect to an image,
    we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as labels.
    """

    def __init__(self, bbox, size, mode="xyxy", infos=None):
        # type: (Optional[Union[torch.Tensor, np.ndarray, Sequence]], Sequence[int], str, Optional[dict])->None
        super().__init__(infos)
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        if bbox is None:
            bbox = torch.empty(0, 4, dtype=torch.float32, device=device)
        elif isinstance(bbox, np.ndarray):
            bbox = torch.from_numpy(bbox).float()
        else:
            bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device).view(-1, 4)
        if bbox.ndimension() != 2:
            raise ValueError("bbox should have 2 dimensions, got {}".format(bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError("last dimension of bbox should have a size of 4, got {}".format(bbox.size(-1)))
        if mode not in ("xyxy", "xywh", 'cxcy'):
            raise ValueError("mode should be 'xyxy' or 'xywh' or 'cxcy'")

        assert len(size) == 2, f"Please give correct shape, not '{size}'"

        self.bbox = bbox  # type: torch.Tensor
        self.size = n_tuple(size, 2)  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
        # self.ToRemove = 1  # considering closed form, i.e., including boundary.
        self.ToRemove = 0  # considering open form i.e (min, max), exclude the points exactly at min or max.
        # NOTE: the value of bbox are x in [0, width], y in [0, height]

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data
        return self

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def clone(self):
        copy_ = BoxList(self.bbox.clone(), self.size, self.mode, deepcopy(self.infos))
        for k, v in self.extra_fields.items():
            if hasattr(v, 'clone'):
                copy_.add_field(k, getattr(v, 'clone')())
            else:
                copy_.add_field(k, deepcopy(v))
        return copy_

    def convert_(self, mode):
        if mode == self.mode:
            return self
        if mode not in ("xyxy", "xywh", "cxcy"):
            raise ValueError("mode should be 'xyxy' or 'xywh', 'cxcy'")
        if self.mode == 'xywh' and mode == "xyxy":
            self.bbox[:, 2] += self.bbox[:, 0] - self.ToRemove
            self.bbox[:, 3] += self.bbox[:, 1] - self.ToRemove
        elif self.mode == "xyxy" and mode == 'xywh':
            self.bbox[:, 2] -= self.bbox[:, 0] - self.ToRemove
            self.bbox[:, 3] -= self.bbox[:, 1] - self.ToRemove
        elif self.mode == "xyxy" and mode == 'cxcy':
            wh = self.bbox[:, 2:] - self.bbox[:, :2]
            self.bbox[:, :2] += 0.5 * wh
            self.bbox[:, 2:] = wh
        elif self.mode == "xywh" and mode == 'cxcy':
            self.bbox[:, :2] += 0.5 * self.bbox[:, 2:]
        elif self.mode == 'cxcy' and mode == "xyxy":
            wh = self.bbox[:, 2:]
            self.bbox[:, :2] -= 0.5 * wh
            self.bbox[:, 2:] = self.bbox[:, :2] + wh
        elif self.mode == "cxcy" and mode == 'xywh':
            self.bbox[:, :2] -= 0.5 * self.bbox[:, 2:]
        self.mode = mode
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self
        else:
            result = self.clone()
            result.convert_(mode)
            return result

    def resize_(self, size, *args, **kwargs):
        """
        Resize this bounding box

        :param size: The requested size in pixels, as a 2-tuple: (width, height).
        """
        self.convert_("xyxy")
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            self.bbox *= ratios[0]
        else:
            self.bbox *= self.bbox.new_tensor([[ratios[0], ratios[1], ratios[0], ratios[1]]])

        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v.resize_(size, *args, **kwargs)
        self.size = size
        return self

    def resize(self, size, *args, **kwargs):
        result = self.clone()
        result.resize_(size, *args, **kwargs)
        return result

    def flip_(self, horizontal=False, vertical=False, *args, **kwargs):
        """
        Transpose bounding box (flip_left_right, flip_top_bottom)
        """
        self.convert_('xyxy')

        if horizontal:
            transposed_xmin = torch.sub(self.size[0] - self.ToRemove, self.bbox[:, 2])
            transposed_xmax = torch.sub(self.size[0] - self.ToRemove, self.bbox[:, 0])
            self.bbox[:, 0] = transposed_xmin
            self.bbox[:, 2] = transposed_xmax
        if vertical:
            transposed_ymin = torch.sub(self.size[1] - self.ToRemove, self.bbox[:, 3])
            transposed_ymax = torch.sub(self.size[1] - self.ToRemove, self.bbox[:, 1])
            self.bbox[:, 1] = transposed_ymin
            self.bbox[:, 3] = transposed_ymax
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v.flip_(horizontal=horizontal, vertical=vertical)
        return self

    def flip(self, horizontal=False, vertical=False, *args, **kwargs):
        return self.clone().flip_(horizontal, vertical, *args, **kwargs)

    def crop_(self, x, y, w, h, remove_area=-1, clamp=True, overlap=False, *args, **kwargs):
        """
        Crop a rectangular region from this bounding box.
        """
        self.convert_("xyxy")
        area = self.area() if remove_area >= 0 or overlap else None
        self.bbox -= self.bbox.new_tensor([[x, y, x, y]])
        if clamp:
            self.bbox[:, 0].clamp_(0, w - self.ToRemove)
            self.bbox[:, 1].clamp_(0, h - self.ToRemove)
            self.bbox[:, 2].clamp_(0, w - self.ToRemove)
            self.bbox[:, 3].clamp_(0, h - self.ToRemove)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v.crop_(x, y, w, h)
        self.size = (w, h)

        if remove_area >= 0:
            new_area = self.area()
            keep = (new_area / area) >= remove_area
            self.bbox = self.bbox[keep]
            self.extra_fields = {k: v[keep] for k, v in self.extra_fields.items()}
        if overlap:
            new_area = self.area()
            self.add_field('overlaps', area / new_area)
        return self

    def crop(self, x, y, w, h, remove_area=-1, clamp=True, overlap=False, *args, **kwargs):
        return self.clone().crop_(x, y, w, h, remove_area=remove_area, clamp=clamp, overlap=overlap)

    def pad_(self, padding, *args, **kwargs):
        """
        Pad some pixels around the bounding box. The order is left(, top, (right, and bottom)).
        """
        left_pad, top_pad, right_pad, bottom_pad = self._get_padding_param(padding)
        if self.mode == "xyxy":
            self.bbox += self.bbox.new_tensor([[left_pad, top_pad, left_pad, top_pad]])
        else:
            self.bbox[:, 0] += left_pad
            self.bbox[:, 1] += top_pad
        w, h = self.size
        w = w + left_pad + right_pad
        h = h + top_pad + bottom_pad
        self.size = (w, h)

        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v.pad_(padding)
        return self

    def pad(self, padding, *args, **kwargs):
        left_pad, top_pad, right_pad, bottom_pad = self._get_padding_param(padding)
        result = BoxList(self.bbox, self.size, self.mode, self.infos)
        if self.mode == "xyxy":
            result.bbox = self.bbox + self.bbox.new_tensor([[left_pad, top_pad, left_pad, top_pad]])
        else:
            result.bbox[:, 0] = self.bbox[: 0] + left_pad
            result.bbox[:, 1] = self.bbox[:, 0] + top_pad
        w, h = self.size
        w = w + left_pad + right_pad
        h = h + top_pad + bottom_pad
        result.size = (w, h)

        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.pad(padding)
            result.extra_fields[k] = v
        return result

    # Tensor-like methods
    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = getattr(v, "to")(device)
            bbox.add_field(k, v)
        bbox.add_info(**self.infos)
        return bbox

    def __getitem__(self, item):
        if self.bbox.numel() == 0:
            return self
        else:
            if isinstance(item, slice):
                bbox = BoxList(self.bbox[item], self.size, self.mode)
                for k, v in self.extra_fields.items():
                    bbox.add_field(k, v[item])
            else:
                if isinstance(item, int):
                    item = [item]
                bbox = BoxList(self.bbox[item] if len(item) else None, self.size, self.mode)
                for k, v in self.extra_fields.items():
                    bbox.add_field(k, v[item] if len(item) else None)
            bbox.add_info(**self.infos)
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        self.convert_('xyxy')
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - self.ToRemove)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - self.ToRemove)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - self.ToRemove)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - self.ToRemove)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v.clip_to_image()
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def remove_empty_(self, remove=True):
        if not remove:
            return self
        self.convert_('xyxy')
        box = self.bbox
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        self.bbox = self.bbox[keep]
        self.extra_fields = {k: v[keep] for k, v in self.extra_fields.items()}
        return self

    def get_keep_mask(self):
        if self.mode == 'xyxy':
            return (self.bbox[:, 3] > self.bbox[:, 1]) & (self.bbox[:, 2] > self.bbox[:, 0])
        else:
            return self.bbox[:, 2:].gt(0).all(dim=1)

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            area = (box[:, 2] - box[:, 0] + self.ToRemove) * (box[:, 3] - box[:, 1] + self.ToRemove)
        elif self.mode == "xywh" or self.mode == 'cxcy':
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return area

    def copy_with_fields(self, fields):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox.add_field(field, self.get_field(field))
        bbox.add_info(**self.infos)
        return bbox

    def intersect(self, o) -> torch.Tensor:
        """
        Compute the intersecting area of two sets of boxes, i.e. A ∩ B
        Args:
            self: Multiple bounding boxes, Shape: [num_boxes,4]
            o: Multiple bounding box, Shape: [num_boxes_o, 4]
        Return:
            intersect: Shape: [num_boxes, num_boxes_o]
        """
        self.convert_('xyxy')
        o.convert_('xyxy')
        max_xy = torch.min(self.bbox[:, None, 2:], o.bbox[None, :, 2:])
        min_xy = torch.max(self.bbox[:, None, :2], o.bbox[None, :, :2])
        inter = (max_xy - min_xy + self.ToRemove).clamp_(0)
        return inter[:, :, 0] * inter[:, :, 1]

    def iou(self, box_b, use_crowd=False) -> torch.Tensor:
        """Compute the Jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            use_crowd: for coco, iscrowd=True, iou=inter/area_det, else iou=inter/union
            self: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [1, 4]
        Return:
            jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self.intersect(box_b)
        area_a = self.area()
        area_b = box_b.area()
        union = area_a[:, None] + area_b[None, :] - inter
        if use_crowd:
            if self.has_field('iscrowd'):
                union = torch.where(self.get_field('iscrowd')[:, None], area_b[None, :], union)
            elif box_b.has_field('iscrowd'):
                union = torch.where(box_b.get_field('iscrowd')[None, :], area_a[:, None], union)
        return inter / union

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "(width, height)={}, ".format(self.size)
        s += "mode={}".format(self.mode)
        if len(self.extra_fields) > 0:
            s += ", field: [" + ", ".join(self.extra_fields.keys()) + "]"
        if len(self.infos) > 0:
            s += ", infos: {}".format(self.infos)
        s += ")"
        return s

    # def __getattr__(self, item):
    #     if item in self.__dict__:
    #         return self.__dict__[item]
    #     elif item in self.extra_fields:
    #         return self.extra_fields[item]
    #     elif item in self.infos:
    #         return self.infos[item]
    #     else:
    #         raise AttributeError(f'No such attribute {item}')

    def nms(self, threshold=0.1, num_classes: int = -1, keep_threshold=1e-3, soft=False, sigma=0.5):
        self.convert_('xyxy')
        assert 'labels' in self.extra_fields and 'scores' in self.extra_fields
        ops.nms(self.bbox, self.get_field('labels'), self.get_field('scores'), threshold=threshold, soft=soft,
            sigma=sigma, num_classes=num_classes, to_remove=self.ToRemove)
        keep = self.get_field('scores') > keep_threshold
        return self[keep]

    def soft_linear_nms(self, threshold=0.1, num_classes: int = -1, keep_threshold=1e-3):
        return self.nms(threshold, num_classes, keep_threshold, True, -1)

    def soft_exp_nms(self, threshold=0.1, num_classes=-1, sigma=0.5, keep_threshold=1e-3):
        return self.nms(threshold, num_classes, keep_threshold, True, sigma)

    def test_transform(self):
        print('Deprecated!', __file__)
        if 'transform' not in self.infos:
            return self
        transform = self.infos.pop('transform')  # remove to avoid repeat transform
        result = self
        for t in reversed(transform):
            result = getattr(result, t[0])(*t[1:])
        return result

    def restore(self, rt: Restoration):
        result = self
        for t in rt.get_transforms():
            result = getattr(result, t[0])(*t[1:-1], **t[-1])
        return result

    def get_affine_matrix(self, angle=0., translate=(0., 0.), scale=1., shear=(0., 0.), center=None, output_size=None):
        """
        center is default as the center of images
        1. move the origin to center
        2. rotate the image <angle> degrees
        3. shear the image
        4. scale
        5. move the origin back
        6. translate
        """
        if center is None:
            cx, cy = (self.size[0] * 0.5, self.size[1] * 0.5)
        else:
            cx, cy = center
        R = np.eye(3)
        angle = math.radians(angle)
        cos_a, sin_a = scale * math.cos(angle), scale * math.sin(angle)
        sx, sy = math.tan(math.radians(shear[0])), math.tan(math.radians(shear[1]))
        a = cos_a + sin_a * sy
        b = cos_a * sx + sin_a
        c = -sin_a + cos_a * sy
        d = -sin_a * sx + cos_a
        R[0] = [a, b, cx + translate[0] - a * cx - b * cy]
        R[1] = [c, d, cy + translate[1] - c * cx - d * cy]
        return R, True

    def affine(self, affine_matrix, output_size, **kwargs):
        n = len(self)
        if n == 0:
            return self
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix)
        assert isinstance(affine_matrix, torch.Tensor)
        self.convert_('xyxy')
        affine_matrix = affine_matrix.to(self.bbox)
        t = torch.ones(n * 4, 3)
        t[:, :2] = self.bbox[:, [0, 1, 0, 3, 2, 1, 2, 3]].view(-1, 2)
        t = t.mm(affine_matrix.T).reshape(n, 4, 3)
        x = t[:, :, 0]
        y = t[:, :, 1]
        bbox = torch.stack([x.min(1)[0], y.min(1)[0], x.max(1)[0], y.max(1)[0]], dim=-1)
        new_bbox = BoxList(bbox, self.size, self.mode, infos=self.infos)
        for key, value in self.extra_fields.items():
            new_bbox.add_field(key, value.affine(affine_matrix, output_size) if hasattr(value, 'affine') else value)
        new_bbox.crop_(0, 0, *output_size)
        return new_bbox

    def draw(
        self, img: np.ndarray = None, class_names=None, det_colors=None, num_classes=None, font_size=None, *args,
        **kwargs
    ):
        bbox = self.convert("xyxy")
        bb = bbox.bbox.detach().cpu().numpy()
        labels = bbox.get_field('labels') if bbox.has_field('labels') else None
        scores = bbox.get_field('scores') if bbox.has_field('scores') else None

        if det_colors is None:
            if class_names is not None:
                det_colors = get_colors(len(class_names))
            elif num_classes is not None:
                det_colors = get_colors(num_classes)
            elif labels is not None:
                det_colors = get_colors(labels.max().item() + 1)
            else:
                det_colors = get_colors(1)

        if font_size is None:
            font_size = int(1.0e-2 * max(bbox.size) + 7)

        img = PIL.Image.fromarray(img)
        draw = PIL.ImageDraw.Draw(img)
        font = PIL.ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size=font_size)
        for i, (x1, y1, x2, y2) in enumerate(bb):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if labels is not None:
                cls = labels[i].item()
                color = tuple(det_colors[cls])
                text = str(cls) if class_names is None else class_names[cls]  # type: str
                if scores is not None:
                    text += f':{scores[i].item():.2f}'
                tw, th = draw.textsize(text, font)
                tx, ty = x1, y1
                if ty >= th:
                    ty -= th
                if tx + tw > bbox.size[0]:
                    tx = max(0, bbox.size[0] - tw)
                draw.rectangle([(tx, ty), (tx + tw, ty + th)], fill=color)
                draw.text((tx, ty), text, font=font, fill=tuple([255 - c for c in color]))
            else:
                color = (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        return np.array(img)


def gather_bounding_boxes(boxes: List[BoxList]):
    """注意 抛弃BoxList中的info"""
    if get_world_size() == 1:
        return boxes
    device = boxes[0].bbox.device
    num_boxes = torch.tensor([len(box) for box in boxes], dtype=torch.long, device=device)
    num_boxes = gather_tensor(num_boxes, 0, is_same_shape=False)

    box_sizes = torch.tensor([box.size for box in boxes], dtype=torch.long, device=device)
    box_sizes = gather_tensor(box_sizes, 0, is_same_shape=False)

    bboxes = torch.cat([box.bbox for box in boxes])
    bboxes = gather_tensor(bboxes, 0, is_same_shape=False)

    extra_fields = {}
    for key in boxes[0].fields():
        assert isinstance(boxes[0].get_field(key), torch.Tensor)
        extra_fields[key] = gather_tensor(torch.cat([box.get_field(key) for box in boxes], 0), 0, is_same_shape=False)

    all_boxes = []
    st = 0
    for i in range(len(box_sizes)):
        ed = st + num_boxes[i].item()
        box = BoxList(bboxes[st:ed], box_sizes[i].tolist(), mode=boxes[0].mode)
        for k, v in extra_fields.items():
            box.add_field(k, v[st:ed])
        all_boxes.append(box)
        st = ed
    return all_boxes
