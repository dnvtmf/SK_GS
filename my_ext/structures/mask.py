from typing import Union

import torch
from torch import Tensor

from my_ext._C import get_C_function, have_C_functions


class Masks:
    """ the structure of segmentation masks

    Args:
        data: shape [..., H, W] or [M] binary or encoded masks
        start: shape None or [..., H];
        shape: None or (..., H, W)
        format: BINARY, 'bin' or ENCODED 'enc'
    """
    BINARY = 'bin'  # binary mask
    ENCODED = 'enc'  # using run-length encoding (RLE), but only encode last dim

    # MULTI = 'multi'  # encode many masks
    eps = 1e-7

    def __init__(self, data: Tensor, start: Tensor = None, shape=None, format=ENCODED) -> None:
        if start is None:
            assert data.dtype == torch.bool
            self.data = data
            self._shape = None
            self.start = None
            self._format = self.BINARY
        else:
            self._shape = torch.Size(shape)
            self.start = start
            self.data = data
            self._format = self.ENCODED
        self._area = None  # cache
        self.format = format

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, f):
        assert f in [self.BINARY, self.ENCODED]
        if f != self._format:
            if self._format == self.BINARY and f == self.ENCODED:
                self.encode_()
            elif self._format == self.ENCODED and f == self.BINARY:
                self.binary_()
            else:
                raise NotImplementedError(f"formant {self.format} --> {f}")

    @property
    def shape(self) -> torch.Size:
        if self._format == self.BINARY:
            return self.data.shape
        else:
            return self._shape

    @property
    def ndim(self):
        return self.data.ndim if self.format == self.BINARY else len(self._shape)

    @property
    def device(self):
        return self.data.device

    def __len__(self):
        return self.shape[0]

    def copy(self):
        """shadow copy"""
        x = Masks(self.data, self.start, self._shape, self._format)
        x._area = self._area
        return x

    def clone(self):
        """deep copy"""
        x = Masks(self.data.clone(), None if self.start is None else self.start.clone(), self._shape, self._format)
        if self._area is not None:
            x._area = self._area.clone()
        return x

    def binary_(self, engine='C'):
        if self.format == self.BINARY:
            return self
        assert self.format == self.ENCODED
        if engine in ['auto', 'C'] and have_C_functions('mask_to_binary'):
            self.data = get_C_function('mask_to_binary')(self._shape[-1], self.start, self.data).view(self._shape)
            self._shape = None
            self.start = None
            self._format = self.BINARY
            return self
        # engine == python
        mask = torch.zeros(self._shape, dtype=torch.bool, device=self.start.device).flatten(0, -2)
        start = self.start.view(-1)
        for i in range(len(start)):
            s = start[i]
            e = len(self.data) if i + 1 == len(start) else start[i + 1]
            n = 0
            for j in range(s + 1, e, 2):
                n = n + self.data[j - 1].item()
                mask[i, n:n + self.data[j].item()] = 1
                n = n + self.data[j].item()
        self.data = mask.view(self._shape)
        self._shape = None
        self.start = None
        self._format = self.BINARY
        return self

    def binary(self, engine='auto'):
        return self.copy().binary_(engine)

    def encode_(self, engine='auto'):
        if self._format == self.ENCODED:
            return self
        assert self._format == self.BINARY
        assert engine in ['python', 'auto', 'C']
        if engine != 'python':
            self._shape = self.data.shape
            self.start, self.data = get_C_function('mask_from_binary')(self.data)
            self.start = self.start.view(self._shape[:-1])
            self._format = self.ENCODED
            return self
        data = self.data
        N = data.shape[-1]
        self._shape = data.shape
        data = torch.constant_pad_nd(data, (1, 1), 0)
        data = torch.constant_pad_nd(data, (1, 1), 1)
        change_index = torch.nonzero(data[..., :-1] != data[..., 1:])
        counts = change_index[1:, -1] - change_index[:-1, -1]
        start = torch.nonzero(counts < 0)[:, 0] - torch.arange(data[..., 0].numel() - 1, device=data.device)
        start = torch.constant_pad_nd(start, (1, 0), 0)
        counts = counts[counts >= 0]
        # assert start.max() < len(counts)
        counts[start] -= 1
        counts[start[1:] - 1] -= 1
        counts[-1] -= 1
        max_value = counts.max()
        if max_value < 256:
            counts = counts.to(torch.uint8)
        elif max_value < (1 << 15):
            counts = counts.to(torch.int16)
        else:
            counts = counts.to(torch.int32)
        start = start.view(data.shape[:-1]).to(torch.int32)
        self.start = start
        self.data = counts
        self._format = self.ENCODED
        return self

    def encode(self, engine='auto') -> 'Masks':
        return self.copy().encode_(engine)

    @property
    def area(self):
        assert self._area is None or self._area.shape == self.shape[:-2]
        if self._area is not None:
            return self._area
        if self._format == self.BINARY:
            self._area = self.data.sum(dim=(-1, -2))
        else:  # self.ENCODED
            # num = self._start_to_num(self.start, self.data)  # DEBUG
            # assert torch.all(num % 2 == 1)  # DEBUG
            start = self.start.view(-1)
            nz = self.data.clone()
            zero_index = torch.ones(len(self.data), device=start.device)
            zero_index[start] = 0
            nz[torch.cumsum(zero_index, dim=0) % 2 == 0] = 0
            sum_nz = torch.cumsum(nz, dim=0)
            index = torch.constant_pad_nd(start, (0, 1), len(self.data))
            # print(sum_nz.shape, index.shape) # DEBUG
            self._area = (sum_nz[index[1:] - 1] - sum_nz[index[:-1]]).view(self._shape[:-1]).sum(dim=-1)
            assert self._area.shape == self._shape[:-2], "DEBUG"
        return self._area

    def intersect(self, other: 'Masks', engine='auto'):
        """|self & other|"""
        H, W = self.shape[-2:]
        assert other.shape[-2:] == self.shape[-2:]
        if self.format == self.BINARY and other.format == self.BINARY:
            masks_a = self.data.reshape(-1, H * W)
            masks_b = other.data.reshape(-1, H * W)
            inter = (masks_a[:, None, :] & masks_b[None, :, :]).sum(dim=-1)
            inter = inter.view(list(self.shape[:-2]) + list(other.shape[:-2]))
            return inter
        A = self.encode()
        B = other.encode()
        area = torch.zeros(list(A.shape[:-2]) + list(B.shape[:-2]), dtype=torch.float, device=self.device)
        if engine in ['auto', 'C'] and have_C_functions('intersect'):
            get_C_function('intersect')(W, A.start, A.data, B.start, B.data, area)
            return area
        ## python engine
        print('Use python engine, please try C engine for speed')
        area_ = area.view(A.start[..., 0].numel(), -1)
        sa = A.start.view(-1, H)
        sb = B.start.view(-1, H)
        for i in range(area_.shape[0]):
            for j in range(area_.shape[1]):
                temp = 0
                for k in range(A.shape[-2]):
                    si, sj = sa[i, k].item(), sb[j, k].item()
                    la, lb = A.data[si].item(), B.data[sj].item()
                    na, nb = la, lb
                    a, b = 0, 0
                    while na < W or nb < W:
                        if na < nb:
                            a ^= 1
                            si += 1
                            la = A.data[si].item()
                            na += la
                        else:
                            b ^= 1
                            sj += 1
                            lb = B.data[sj].item()
                            nb += lb
                        if a == 1 and b == 1:
                            temp += min(na, nb) - max(na - la, nb - lb)
                area_[i, j] = temp
        return area

    def __and__(self, other: 'Masks'):
        return self.intersect(other)

    def In(self, other: 'Masks'):
        """ |self & other| / |self|"""
        inter = self.intersect(other)
        area = self.area + self.eps
        return (inter.view(area.numel(), -1) / area.view(-1, 1)).view_as(inter)

    def IoU(self, other: 'Masks'):
        inter = self.intersect(other)
        shape = inter.shape
        area = self.area.view(-1, 1)
        area_o = other.area.view(1, -1)
        inter = inter.view(area.numel(), -1)
        return (inter / (area + area_o - inter + self.eps)).view(shape)

    def cpu(self):
        if self.start is not None:
            self.start = self.start.cpu()
        self.data = self.data.cpu()
        if self._area is not None:
            self._area = self._area.cpu()
        return self

    def cuda(self):
        if self.start is not None:
            self.start = self.start.cuda()
        self.data = self.data.cuda()
        if self._area is not None:
            self._area = self._area.cuda()
        return self

    def to(self, device=None, **kwargs):
        if self.start is not None:
            self.start = self.start.to(device=device, **kwargs)
        self.data = self.data.to(device=device, **kwargs)
        if self._area is not None:
            self._area = self._area.to(device=device, **kwargs)
        return self

    def cat(self, other: 'Masks'):
        """cat along dim 0"""
        if self.format == self.BINARY and other.format == self.BINARY:
            res = Masks(torch.cat([self.data, other.data], dim=0), format=self.BINARY)
        else:
            A = self.encode()
            B = other.encode()
            assert len(A._shape) == len(B._shape) and A.shape[1:] == B.shape[1:]
            shape = list(A.shape)
            shape[0] += B.shape[0]
            start = torch.cat([A.start, B.start + len(A.data)], dim=0)
            data = torch.cat([A.data, B.data], dim=0)
            res = Masks(data, start, tuple(shape), format=self.ENCODED)
        if self._area is not None and other._area is not None:
            res._area = torch.cat([self._area, other._area], dim=0)
        return res

    def __add__(self, other: 'Masks') -> 'Masks':
        return self.cat(other)

    def squeeze_(self, *dim: int):
        if self.format == self.BINARY:
            self.data.squeeze_(*dim)
        else:
            self.start.squeeze_(*dim)
            self._shape = torch.Size(list(self.start.shape) + [self._shape[-1]])
        if self._area is not None:
            self._area.squeeze_(*dim)
        return self

    def unsqueeze_(self, dim: int):
        if self.format == self.BINARY:
            self.data.unsqueeze_(dim)
        else:
            self.start.unsqueeze_(dim)
            self._shape = torch.Size(list(self.start.shape) + [self._shape[-1]])
        if self._area is not None:
            self._area.unsqueeze_(dim)
        return self

    def squeeze(self, *dim: int):
        return self.copy().squeeze_(*dim)

    def unsqueeze(self, dim: int):
        return self.copy().unsqueeze_(dim)

    @staticmethod
    def _start_to_num(start: Tensor, data: Tensor):
        shape = start.shape
        start = start.view(-1)
        start = torch.constant_pad_nd(start, (0, 1), len(data))
        num = start[1:] - start[:-1]
        return num.reshape(shape)

    @staticmethod
    def _num_to_start(num: Tensor) -> Tensor:
        return (torch.cumsum(num.view(-1), dim=0).view(num.shape) - num).int()

    def __getitem__(self, index) -> "Masks":
        if self.format == self.BINARY:
            out = Masks(self.data[index], format=self.BINARY)
            if self._area is not None:
                out._area = self._area[index]
            return out
        # format = self.ENCODED
        if isinstance(index, slice):
            s, e, t = index.indices(self._shape[0])
            shape = torch.Size([e - s] + list(self._shape[1:]))
            if s >= e:
                return Masks(self.data[:0], self.start[:0], shape, self.format)
            if t == 1 and s == 0:
                if e == self._shape[0]:
                    return self
                else:
                    N = min(e, self._shape[0])
                    if N == self._shape[0]:
                        return self
                    data = self.data[:self.start[N].view(-1)[0]]
                    out = Masks(data, self.start[:N], [N] + list(self._shape[1:]), self.format)
                    if self._area is not None:
                        out._area = self._area[:N]
                    return out
            else:
                index = torch.arange(s, e, t, device=self.start.device)
        if isinstance(index, Tensor):
            assert index.ndim == 1
            if index.numel() == 0:
                shape = [0] + list(self._shape[1:])
                out = Masks(self.data.new_zeros(0), self.start[:0], shape, format=self.format)
                if self._area is not None:
                    out._area = self._area[:0]
                return out
            N = self.start[0].numel()
            M = self.start.shape[0]
            start = self.start.view(-1, N)
            num = self._start_to_num(start, self.data)
            data = []
            if index.dtype == torch.bool:
                assert index.numel() == M, f"{index.numel()} vs. {M}"
                for out in range(M):
                    if index[out]:
                        data.append(self.data[start[out, 0]:start[out, -1] + num[out, -1]])
            else:
                for out in index:
                    out = out if out >= 0 else out + M
                    assert 0 <= out < M
                    data.append(self.data[start[out, 0]:start[out, -1] + num[out, -1]])
            data = torch.cat(data, dim=0)
            start = self._num_to_start(num[index])
            shape = list(start.shape) + [self._shape[-1]]
            out = Masks(data, start.view(shape[:-1]), shape, format=self.format)
            if self._area is not None:
                out._area = self._area[index]
            return out
        elif index is None:
            return self.unsqueeze(dim=0)
        else:
            assert isinstance(index, int)
            N = self._shape[0]
            index = index if index >= 0 else index + N
            assert 0 <= index < N
            start = self.start.view(N, -1)
            s = self.start[index, 0]
            e = self.start[index + 1, 0] if index + 1 < N else len(self.data)
            data = self.data[s:e]
            out = Masks(data, self.start[index] - s, self._shape[1:], format=self.format)
            if self._area is not None:
                out._area = self._area[index]
            return out

    def __setitem__(self, index, value: Union[Tensor, 'Masks']):
        if isinstance(value, Tensor):
            value = Masks(value, format=self.format)
        if self.format == self.BINARY:
            self.data[index] = value.data
        elif isinstance(index, int):
            index = index if index >= 0 else index + self._shape[0]
            s = self.start[index].view(-1)[0].item()
            e = len(self.data) if index + 1 == self._shape[0] else self.start[index + 1].view(-1)[0].item()
            self.start[index] = value.start + s
            self.start[index + 1:] += len(value.data) - (e - s)
            self.data = torch.cat([self.data[:s], value.data, self.data[e:]])
        elif isinstance(index, slice):
            start, stop, step = index.indices(self._shape[0])
            assert step == 1
            s = self.start[start].view(-1)[0].item()
            e = len(self.data) if stop == self._shape[0] else self.start[stop].view(-1)[0].item()
            self.start[start:stop] = value.start + s
            self.start[stop:] += len(value.data) - (e - s)
            self.data = torch.cat([self.data[:s], value.data, self.data[e:]])
        else:
            raise ValueError(f"Not support index type: {type(index)}")
        if self._area is not None:
            self._area[index] = value.area

    def __repr__(self):
        return f"Masks[{self.format}]({list(self.shape)})"

    def pad(self, before=0, after=0, dim=0):
        if before == 0 and after == 0:
            return self
        dim = dim if dim >= 0 else dim + self.ndim
        assert 0 <= dim < self.ndim
        if self.format == self.BINARY:
            pad = [0] * (2 * self.data.ndim)
            dim_ = self.ndim - 1 - dim
            pad[2 * dim_] = before
            pad[2 * dim_ + 1] = after
            data = torch.constant_pad_nd(self.data, pad)
            out = Masks(data, format=self.format)
        else:
            assert dim == 0 and before == 0
            W = self._shape[-1]
            new_data = []
            if self.data.dtype == torch.uint8:
                while W > 255:
                    new_data.append(255)
                    new_data.append(0)
                    W -= 255
                new_data.append(W)
            N = after * self.start[0].numel()
            new_data = self.data.new_tensor([new_data]).expand(N, -1).contiguous()
            new_start = torch.arange(N).to(self.start) * new_data.shape[-1] + len(self.data)
            shape = [self._shape[0] + after] + list(self._shape[1:])
            start = torch.cat([self.start.view(-1), new_start], dim=0).view(shape[:-1])
            data = torch.cat([self.data, new_data.view(-1)], dim=0)
            out = Masks(data, start, shape, format=self.format)
        if self._area is not None:
            pad = [0] * (2 * self._area.ndim)
            dim_ = self._area.ndim - 1 - dim
            pad[2 * dim_] = before
            pad[2 * dim_ + 1] = after
            out._area = torch.constant_pad_nd(self._area, pad)
        return out


def test():
    a = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [0, 1, 0, 1, 0]],
        dtype=torch.bool)
    print(a.int())
    print(a.shape)
    d = Masks(a, format=Masks.BINARY)
    b = d.encode(engine='python')
    print(b)
    f = d.encode(engine='C')
    print(b.shape, f.shape)
    print(b.start, f.start)
    print(b.data, f.data)
    assert (b.shape == f.shape and b.start.eq(f.start).all() and b.data.eq(f.data).all())
    assert (d.data.eq(f.binary().data).all())
    e = d.cuda().encode(engine='C').cpu()
    d.cpu()
    assert (b.shape == e.shape and b.start.eq(e.start).all() and b.data.eq(e.data).all())
    assert (d.data.eq(e.binary().data).all())

    print(b.binary())
    ## test binary
    assert (b.binary(engine='python').data == a).all()
    assert (b.binary(engine='C').data == a).all()
    print(b.area.shape, b.area, a.sum())
    print('cpu:', b.intersect(b, engine='C'))
    b.cuda()
    assert (b.binary(engine='C').cpu().data == a).all()
    ## test intersect
    print('cuda:', b.intersect(b, engine='C'))
    print('python:', b.intersect(b, engine='python'))
    print(b.In(b))
    print(b.IoU(b))

    b.unsqueeze_(0)
    print(b)
    print(b.binary().data.int())
    c = b + b
    print(c.binary().data.int())
    assert (b.start - b._num_to_start(b._start_to_num(b.start, b.data))).abs().sum().item() == 0, 'convert start'
    print(b._start_to_num(b.start, b.data))
    assert (c[0].binary().data.cpu() == a).all()
    print(c[:10])
    print(c[-1:])
    assert c[torch.tensor([False, True])].binary().data[0].cpu().eq(a).all()
    assert c[torch.tensor([1])].binary().data[0].cpu().eq(a).all()
    print(d[None].pad(0, 1).data.int())
    e = c.pad(0, 1)
    print(e)
    print(e.binary().data.int())
    e[-1] = b[0]
    print(e.binary().data.int())


def test_empty():
    a = torch.zeros((0, 100), dtype=torch.bool).cuda()
    print(a)
    b = Masks(a, format=Masks.BINARY)
    print(b)
    c = b.encode()
    print(c)
    d = c.binary()
    print(d)
    # index
    e = c[:1]
    print(e)
    e = d[:1]
    print(e)
    index = torch.tensor([], dtype=torch.long, device='cuda')
    print(index)
    print(c[index])
    print(d[index])


if __name__ == '__main__':
    test()
