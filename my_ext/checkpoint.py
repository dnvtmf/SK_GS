import os
from pathlib import Path
from typing import List
import logging

import torch
from torch import nn

import my_ext as ext
from my_ext.config import get_parser
from my_ext.utils import add_bool_option, add_path_option, add_cfg_option, add_dict_option

__all__ = ['options', 'make', 'CheckpointManager']


def options(parser=None):
    group = get_parser(parser).add_argument_group("Save Options")
    # for checkpoint
    add_path_option(group, "--resume", default=None, help="path to the checkpoint needed resume")
    group.add_argument(
        "--checkpoint-interval",
        default=1,
        type=int,
        metavar='N',
        help="The interval epochs for saving checkpoint. (never save when <= 0)"
    )
    group.add_argument(
        "--checkpoint-max-keep",
        default=1,
        type=int,
        metavar='N',
        help="The maximum number of checkpoint kept on disk."
    )
    add_dict_option(group, '--checkpoint-save', default={},
        help='save checkpoint using given name at given epoch, format: {epoch: name, epoch: \'\'}')
    # for trained model
    add_path_option(group, "--load", default=None, help="The path to (pre-)trained model.")
    add_cfg_option(group, '--load-cfg', default={}, help='The configuare when load (pre-)trained model')
    add_bool_option(
        group,
        "--load-no-strict",
        default=False,
        help="The keys of loaded model may not exactly match the model's. (May usefully for finetune)"
    )
    return group


def make(cfg):
    # checkpoint loaded in config.make -> _load_from_pth
    return CheckpointManager(
        checkpoint_interval=cfg.checkpoint_interval,
        num_checkpoint_max=cfg.checkpoint_max_keep,
        save_at=cfg.checkpoint_save,
    )


class CheckpointManager:
    _loaded_checkpoint = {}

    def __init__(self, checkpoint_interval=1, num_checkpoint_max=1, save_at: dict = None):
        self._store_objects = {}  # the objects which are need to save to checkpoint
        self._saved_file_paths = []  # type: List[Path]
        self._save_dir = Path('.')

        self.prefix = 'checkpoint'
        self.suffix = '.pth'

        self.checkpoint_interval = checkpoint_interval
        self.num_checkpoint_max = num_checkpoint_max  # The maximum number of checkpoint
        self.save_at = {} if save_at is None else {int(k): v if v else f"checkpoint_{k}" for k, v in save_at.items()}

    @staticmethod
    def load_checkpoint(file_path):
        if not file_path:
            return
        if os.path.exists(file_path):
            CheckpointManager._loaded_checkpoint = torch.load(file_path, map_location='cpu')
            logging.info(f"Load checkpoint from: {file_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file {file_path} is not existed!!!")

    @staticmethod
    def resume(name, default=None):
        return CheckpointManager._loaded_checkpoint.pop(name, default)

    def store(self, name: str, cls: object, attr: str = None):
        # when save checkpoint, the attribute <attr/name> of object <cls> will be stored
        # If checkpoint is loaded, the loaded attribute will be resumed to object <cls>
        assert name not in self._store_objects, f"Duplicate Name `{name}` in checkpoint!"
        attr = name if attr is None else attr
        self._store_objects[name] = (cls, attr)
        # resume
        if name not in self._loaded_checkpoint:
            return
        value = self.resume(name)
        if hasattr(getattr(cls, attr), "load_state_dict"):  # value is not None and
            obj = getattr(cls, attr)
            if isinstance(obj, nn.Module):
                value = ext.utils.state_dict_strip_prefix_if_present(value, "module.")
            if isinstance(obj, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
                obj.module.load_state_dict(value)
            else:
                obj.load_state_dict(value)
            logging.info(f"==> Load state_dict `{name}` from checkpoint")
        else:
            setattr(cls, attr, value)
            logging.info(f"==> Resume `{name}`={ext.utils.show_shape(value)}")
        return

    def set_save_dir(self, save_dir: Path):
        self._save_dir = save_dir

    def save(self, filename: str = '', epoch=-1, save_dir: Path = None, use_prefix=True, manage=True, **kwargs):
        """ save a checkpoint
        Args:
            filename: The filename of saved checkpoint, auto rename to 'checkpoint*.pth'
            epoch: add epoch info when num_checkpoint_max > 1 and filename == ''
            save_dir: Choose another directory to save checkpoint
            use_prefix: The filename of checkpoint must start with 'checkpoint'?
            manage: whether to auto remove checkpoint when the number of checkpoint > num_checkpoint_max
            kwargs: Another data want to save to checkpoint
        """
        if not ext.is_main_process():
            return
        if not ext.utils.check_interval(epoch, self.checkpoint_interval) and epoch not in self.save_at:
            return
        data = kwargs
        for name, (cls, attr) in self._store_objects.items():
            value = getattr(cls, attr)
            if hasattr(value, "state_dict"):
                value = value.state_dict()
                if isinstance(value, torch.nn.Module):
                    value = ext.utils.state_dict_strip_prefix_if_present(value.state_dict(), "module.")
            data[name] = value

        if epoch in self.save_at:
            save_path = (self._save_dir if save_dir is None else save_dir).joinpath(self.save_at[epoch])
            save_path = save_path.with_suffix(self.suffix)
            torch.save(data, save_path)
            logging.info(f"Save checkpoint to {save_path}")
        if not ext.utils.check_interval(epoch, self.checkpoint_interval):
            return
        if use_prefix and filename.startswith(self.prefix):
            filename = filename[len(self.prefix):]
        if filename.endswith(self.suffix):
            filename = filename[:-len(self.suffix)]
        if not filename and self.num_checkpoint_max > 1:
            filename = f'_{epoch}'
        save_path = (self._save_dir if save_dir is None else save_dir).joinpath(self.prefix + filename + self.suffix)

        torch.save(data, save_path)
        logging.info(f"Save checkpoint to {save_path}")

        if manage:
            if save_path in self._saved_file_paths:
                self._saved_file_paths.remove(save_path)
            self._saved_file_paths.append(save_path)
            if len(self._saved_file_paths) > self.num_checkpoint_max:
                self.remove_oldest()
        return

    def remove_all(self):
        cnt = 0
        for file_path in self._saved_file_paths:
            if file_path.exists():
                os.remove(file_path)
                cnt += 1
        logging.info(f'Removed all checkpoints ({cnt})')

    def remove_oldest(self):
        # remove oldest checkpoint
        if len(self._saved_file_paths) == 0:
            return
        file_path = self._saved_file_paths.pop(0)
        if file_path.exists():
            os.remove(file_path)
            logging.info(f'Remove oldest checkpoint: {file_path}')

    def remove_all_other_train(self, prefix='checkpoint', suffix='.pth'):
        # remove all files in save_dir, which filename have <prefix> and <suffix>
        for filename in self._save_dir.iterdir():
            name = filename.name
            if name.startswith(prefix) and name.endswith(suffix):
                os.remove(filename)
