import argparse
import re
from typing import Optional
import logging

import torch
from torch import nn
from torch.nn.modules.batchnorm import _NormBase

from my_ext.config import get_parser
from my_ext.utils import str2dict, eval_str, add_bool_option
from my_ext.optimizer import OPTIMIZERS
from my_ext.utils.torch_utils import get_net
from .SAM import SAM


def options(parser=None):
    group = get_parser(parser).add_argument_group("Optimizer Options:")
    group.add_argument("-oo", "--optimizer", default="sgd", choices=OPTIMIZERS.keys(), metavar='C',
        help="the optimizer method to train network {" + ", ".join(OPTIMIZERS.keys()) + "}")
    group.add_argument("-oc", "--optimizer-cfg", default={}, type=str2dict, metavar="D",
        help="The configure for optimizer")
    group.add_argument("-wd", "--weight-decay", default=0, type=float, metavar="V",
        help="weight decay (default: 0.).")
    add_bool_option(group, "--no-wd-bias", default=False, help="Set the weight decay on bias and bn to 0")
    group.add_argument('--optimizer-groups', default=None, type=eval_str, metavar='D',
        help='将模型参数划分为不同组并使用不同的参数, 当lr=0时, 参数不更新, 当lr<0时, 参数不更新且BN被冻结'
             '格式: {"backbone.*": {lr: 0, weight_decay: 0}, ...} "backbone.*"表示正则表达式')
    add_bool_option(group, "--optimizer-sam", default=False, help='Use SAM optimizer')
    group.add_argument('--optimizer-sam-cfg', default={}, type=str2dict, metavar='D',
        help="The configure SAM optimizer, default: {rho=0.05, adaptive}")
    return


def freeze_modules(cfg, model: nn.Module):
    if getattr(cfg, 'optimizer_groups', None) is None:
        return
    # freeze learnable parameters
    num_frozen_params = 0
    patterns = [(re.compile(k), v.get('lr', 1.) <= 0) for k, v in cfg.optimizer_groups.items()]
    for name, param in get_net(model).named_parameters():  # type: str, nn.Module
        for pattern, is_frozen in patterns:
            if pattern.fullmatch(name) is not None:
                if is_frozen:
                    param.requires_grad_(False)
                    num_frozen_params += 1
                break
    # freeze BN
    patterns = [re.compile(k) for k, v in cfg.optimizer_groups.items() if v.get('lr', 1.) < 0]
    num_frozen_bn = 0
    for name, m in get_net(model).named_modules():  # type: str, nn.Module
        if isinstance(m, _NormBase):
            for pattern in patterns:
                if pattern.fullmatch(name + '.weight') is not None:
                    m.freeze()
                    num_frozen_bn += 1
                    break
    if num_frozen_bn or num_frozen_params:
        logging.info(f"There are {num_frozen_bn} frozen BatchNorm, {num_frozen_params} frozen parameters.")
    return


def _add_weight_decay(net, l2_value, skip_list=()):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': decay, 'weight_decay': l2_value}, {'params': no_decay, 'weight_decay': 0.}, ]


def make(model: Optional[torch.nn.Module], cfg: argparse.Namespace, params=None, **kwargs):
    kwargs = {**cfg.optimizer_cfg, **kwargs}
    if cfg.optimizer == "sgd":
        kwargs.setdefault("momentum", 0.9)
    base_lr = kwargs.setdefault('lr', cfg.lr)
    kwargs.setdefault("weight_decay", cfg.weight_decay)
    if params is None:
        if cfg.optimizer_groups is not None:
            assert isinstance(cfg.optimizer_groups, dict)
            params = [{'params': [], 're': re.compile(f), **_cfg} for f, _cfg in cfg.optimizer_groups.items()]
            params.append({'params': [], 're': re.compile('.*')})
            for name, param in get_net(model).named_parameters():
                for group in params:
                    if group['re'].fullmatch(name) is not None:
                        group['params'].append(param)
                        break
            for (re_s, _cfg), group in zip(cfg.optimizer_groups.items(), params):
                logging.info('Pattern "{}" have {} parameters: {}'.format(re_s,
                    len(group["params"]), ', '.join(f"{k}={v}" for k, v in _cfg.items())))
            if len(params[-1]['params']):
                logging.info(f"Pattern \".*\"  have {len(params[-1]['params'])} parameters.")
            # clean param:
            for group in params:
                group.pop('re')
                group['params'] = [param for param in group['params'] if param.requires_grad]
                group['lr'] = group.get('lr', 1.) * base_lr
            params = [group for group in params if len(group['params'])]
        elif cfg.no_wd_bias:
            params = _add_weight_decay(model, cfg.weight_decay)
        else:
            params = [param for param in model.parameters() if param.requires_grad]
    optimizer = OPTIMIZERS[kwargs.pop('optimizer') if 'optimizer' in kwargs else cfg.optimizer]
    kwargs = {k: v for k, v in kwargs.items() if k in optimizer.__init__.__code__.co_varnames}  # 去掉多余参数
    optimizer = optimizer(params, **kwargs)
    logging.info("==> Optimizer {}".format(optimizer))
    if cfg.optimizer_sam:
        return SAM(optimizer, **cfg.optimizer_sam_cfg)
    else:
        return optimizer
