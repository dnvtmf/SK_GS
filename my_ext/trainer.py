import argparse
import os
import random
import time
import logging

from packaging import version
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.optim.lr_scheduler
import torch.utils.data

import my_ext as ext
from my_ext.config import get_parser
from my_ext.utils import add_bool_option, add_path_option, str2dict
from my_ext.distributed import get_world_size, get_rank


def options(parser=None):
    group = get_parser(parser).add_argument_group("Train Options")
    group.add_argument("-n", "--epochs", default=90, type=int, metavar="N", help="The total number of training epochs.")
    group.add_argument("--nominal-batch-size", default=-1, type=int, metavar="N",
        help="The nominal batch size used for training implemented by gradient accumulation")
    group.add_argument("--start-epoch", default=None, type=int, metavar="N",
        help="manual epoch number (useful on restarts)")
    add_path_option(group, "-o", "--output", default="./results",
        help="The root path to store results (default ./results)")
    add_path_option(group, "--output-dir", default=None,
        help="The actual path to store results (mutually exclusive with output)")
    add_bool_option(group, "--test", default=False, help="Only test model on test set?")
    add_bool_option(group, "--eval", default=False, help="Only test model on validation set?")
    group.add_argument('--eval-interval', default=1, type=int, metavar='N',
        help="Evaluate the model every some epochs. (<= 0 means never eval)")
    group.add_argument("--seed", default=-1, type=int, metavar='N', help="manual seed")
    add_bool_option(group, "--deterministic", default=False, help="use deterministic algorithms for REPRODUCIBILITY")
    add_bool_option(group, "--fp16", default=False, help="Use mixed-precision to train network")
    group.add_argument("--grad-clip", default=-1, type=float, metavar='V',
        help="The value of max norm when perform gradient clip (>0)")
    # group.add_argument("--amp-cfg", default=dict(opt_level='O1', min_loss_scale=1e-3, verbosity=0), type=str2dict,
    #                    metavar='D', help='The configure for amp.initialize')
    add_bool_option(group, '--model-average', default=False, help="Average the model by using SWA/EMA?")
    group.add_argument('--model-average-cfg', default={}, type=str2dict, metavar='D',
        help="The configure, keys: {decay, step, epoch, start_epoch, update_bn}")
    return group


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if version.parse(torch.__version__) >= version.parse('1.8'):
            torch.use_deterministic_algorithms(True)
        else:
            torch.set_deterministic(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 在cuda 10.2及以上的版本中 设置以下变量保证cuda的确定性：
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return


def make(cfg: argparse.Namespace):
    cudnn.benchmark = not cfg.test

    logging.info("==> Config:")
    logging.info(ext.config.save(cfg))

    if not hasattr(cfg, "seed") or cfg.seed < 0:
        cfg.seed = int(time.time() * 100)
        if get_world_size() > 1:  # let seed be same
            seed_tensor = torch.tensor(cfg.seed).cuda()
            torch.distributed.broadcast(seed_tensor, src=0)
            cfg.seed = seed_tensor.item()
    cfg.seed = (cfg.seed + get_rank()) % 2 ** 31  # let seed be different
    set_random_seed(cfg.seed, cfg.deterministic)
    logging.info("==> seed: {}".format(cfg.seed))
    logging.info("==> PyTorch version: {}, cudnn version: {}, CUDA compute capability: {}.{}".format(
        torch.__version__, cudnn.version(), *torch.cuda.get_device_capability()))
    git_version = os.popen("git log --pretty=oneline | head -n 1").readline()[:-1]
    logging.info("==> git version: {}".format(git_version))
    logging.info("==> world size: {}".format(get_world_size()))
    if cfg.fp16:
        logging.info("==> Use mix-precision training")
    if cfg.grad_clip > 0:
        logging.info("==> the max norm of gradient clip is {}".format(cfg.grad_clip))
    if cfg.debug:
        torch.set_anomaly_enabled(True)
        logging.info('==> torch.set_anomaly_enabled')
    return


def model_apply(net: torch.nn.Module, func_name: str, *args, **kwargs):
    def apply_func(m):
        if hasattr(m, func_name):
            getattr(m, func_name)(*args, **kwargs)

    net.apply(apply_func)


def change_with_training_progress(net: torch.nn.Module, step=0, num_steps=1, epoch=0, num_epochs=1):
    """
    change the network status with training progress (current: epoch * num_steps + step, total: num_steps * num_epochs)
    It need module have function change_with_training_progress

    Args:
        net: the module need to change
        step: [0, num_steps - 1], the current step in current epoch
        num_steps: the total number of step for one epoch
        epoch: [0, num_epochs - 1], the current epoch
        num_epochs: the total number of training epochs

    Returns:

    """
    model_apply(net, 'change_with_training_progress', step, num_steps, epoch, num_epochs)
