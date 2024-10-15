import argparse
import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Union, Sequence

import cv2
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader

import my_ext as ext

try:
    import apex
    import apex.parallel

    amp = apex.amp
except ImportError:
    apex = None
    amp = None

cv2.setNumThreads(0)  # Prevent OpenCV from multithreading (to use PyTorch DataLoader)
if platform.system() != 'Windows':
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


# import torch.multiprocessing
#
# torch.multiprocessing.set_sharing_strategy('file_system')


class BaseFramework:
    """A framework for pytorch model train, evaluation, test"""

    def __init__(
        self,
        config_args=None,
        m_data_transforms=None,
        m_datasets=None,
        m_data_loader=None,
        *args,
        **kwargs
    ):
        # module
        self._m_data_transforms = m_data_transforms
        self._m_datasets = m_datasets
        self._m_data_loader = m_data_loader
        #
        self.description = self.__doc__
        self.model_name = ""
        self.model_cfg = ""
        self.output = Path(__file__).absolute().parent.parent / 'results'
        self.device = None  # type: Optional[torch.device]
        self.model = None  # type: Optional[torch.nn.Module]
        self.criterion = None  # type: Optional[torch.nn.Module]
        self.optimizer = None  # type: Optional[Union[torch.optim.SGD, Dict[str,torch.optim.SGD]]]
        self.lr_scheduler = None  # type: Optional[ext.lr_scheduler.LRScheduler]
        self.train_db = None  # type: Optional[Dataset]
        self.eval_db = None  # type: Optional[Dataset]
        self.test_db = None  # type: Optional[Dataset]
        self.train_loader = None  # type: Optional[DataLoader]
        self.eval_loader = None  # type: Optional[DataLoader]
        self.test_loader = None  # type: Optional[DataLoader]
        self.cfg = None  # type:Optional[argparse.Namespace]
        self.logger = None  # type: Optional[logging.Logger]
        self.train_timer = ext.utils.TimeEstimator()  # 运行时间估计
        # self.vis = ext.visualize.Visualization()
        self.amp_scaler = torch.cuda.amp.GradScaler() if torch.__version__ >= '1.6.0' else None
        # self.amp = amp if self.amp_scaler is None else None  # type: amp
        self.checkpoint_manager = ext.checkpoint.CheckpointManager()
        self.interval_grad_acc = 1  # The interval step for gradient accumulation
        # running status
        self.global_step = 0
        self._num_accumulated_steps = 0
        self.epoch = 0
        self.step = 0
        self.num_steps = 1
        self.num_epochs = 1
        self.is_during_training = False
        self.metric_manager = ext.metrics.MetricManager()
        # hooks
        self.hook_manager = ext.utils.HookManager()

    def configure(self, config_args=None, *args, **kwargs):
        self.step_1_config(
            args=config_args,
            ignore_unknown=kwargs.setdefault('ignore_unknown', False),
            ignore_warning=kwargs.setdefault('ignore_warning', False),
            parser=kwargs.setdefault('parser', None),
        )
        self.step_2_environment()
        self.step_3_dataset()
        self.step_4_model()
        self.step_5_data_loader_and_transform()
        self.step_6_optimizer()
        self.step_7_lr()
        self.step_8_others()

    def step_1_config(self, args=None, ignore_unknown=False, ignore_warning=False, parser=None):
        parser = ext.get_parser(parser, self.description)
        # config
        ext.config.options(parser)
        # environment
        ext.checkpoint.options(parser)
        ext.distributed.options(parser)
        ext.trainer.options(parser)
        ext.my_logger.options(parser)
        # ext.visualize.options(parser)
        ext.metrics.options(parser)
        # data
        if self._m_datasets is not None:
            self._m_datasets.options(parser)
        if self._m_data_transforms is not None:
            self._m_data_transforms.options(parser)
        if self._m_data_loader is not None:
            self._m_data_loader.options(parser)
        # model
        # ext.quantization.options(parser)
        # ext.norm_weight.options(parser)
        # solver
        ext.lr_scheduler.options(parser)
        ext.optimizer.options(parser)
        # extra
        self.extra_config(parser)
        self.cfg = ext.config.make(args, ignore_unknown, ignore_warning, parser=parser)
        return self.cfg

    def extra_config(self, parser: argparse.ArgumentParser):
        pass

    def step_2_environment(self, *args, output_paths=(), log_filename=None, **kwargs):
        cfg = self.cfg
        self.checkpoint_manager = ext.checkpoint.make(self.cfg)
        self.store('cfg')
        self.device = ext.distributed.make(cfg)
        self.set_output_dir(*output_paths)
        if log_filename is None:
            log_filename = self.cfg.log_filename if self.cfg.log_filename else (self.mode + ".log")
        self.logger = ext.my_logger.make(cfg, self.output, log_filename, enable=ext.distributed.is_main_process())
        ext.trainer.make(cfg)
        self.logger.info(f"==> the output path: {self.output}")
        self.logger.info(f'==> Task Name: {self.__class__.__name__}')

        self.store('global_step')
        if self.cfg.start_epoch is not None:
            self.epoch = self.cfg.start_epoch

        if self.cfg.fp16 and self.amp_scaler is None:
            self.cfg.fp16 = False
            self.logger.error('Install torch >= 1.6.0 for mixed-precision training.')
        if hasattr(self.cfg, 'batch_size'):
            total_batch_size = ext.get_world_size() * self.cfg.batch_size[0]
            if cfg.reference_lr > 0:
                cfg.lr = self.cfg.reference_lr * total_batch_size / self.cfg.reference_batch_size
                self.logger.info(
                    f'lr={cfg.lr} calculated by reference lr, i.e., '
                    f'{self.cfg.reference_lr} * {total_batch_size} / {self.cfg.reference_batch_size}'
                )
            if self.cfg.nominal_batch_size > 0:
                self.interval_grad_acc = max(1, round(self.cfg.nominal_batch_size / total_batch_size))
                self.logger.info(f'The interval of gradient accumulate: {self.interval_grad_acc}')
        else:
            assert self.cfg.reference_lr <= 0 and self.cfg.nominal_batch_size <= 0

        self.metric_manager = ext.metrics.make(self.cfg)
        self.store('metric_manager')
        self.logger.info(f'==> Metric: {self.metric_manager}')
        self.hook_manager.add_hook(self.metric_manager.summarize, 'after_eval_epoch')
        self.hook_manager.add_hook(self.metric_manager.reset, 'before_eval_epoch')
        self.extra_environment(*args, **kwargs)
        return

    def extra_environment(self, *args, **kwargs):
        pass

    def get_common_model_cfg(self):
        # model_cfg = ext.quantization.make(self.cfg)
        # model_cfg = ext.norm_weight.make(self.cfg)
        return ''

    def step_3_dataset(self, *args, **kwargs):
        if self._m_datasets is None:
            return
        if self.mode == 'train':
            self.train_db = self._m_datasets.make(self.cfg, mode='train')
        if self.mode == 'eval' or (self.mode == 'train' and self.cfg.eval_interval > 0):
            self.eval_db = self._m_datasets.make(self.cfg, mode='eval')
        if self.mode == 'test':
            self.test_db = self._m_datasets.make(self.cfg, mode='test')
        return

    def step_4_model(self, *args, **kwargs):
        pass
        # self.set_output_dir()
        # self.model_name =
        # self.model_cfg = get_common_model_cfg
        # self.model =
        # self.load_model()
        # self.store("model")
        # self.criterion =
        # self.to_cuda()

    def step_5_data_loader_and_transform(self):
        pass

    def step_6_optimizer(self, *args, **kwargs):
        if self.mode != 'train':
            return
        self.optimizer = ext.optimizer.make(self.model, self.cfg)
        self.store("optimizer")
        return

    def step_7_lr(self, *args, **kwargs):
        pass

    def step_8_others(self, *args, **kwargs):
        self.enable_fp16_training()
        self.enable_train_time_estimate()
        self.enable_model_hooks(self.model)

    def enable_visualization(self, env_name):
        # self.vis = ext.visualize.make(self.cfg, env_name, enable=self.mode == 'train', log_dir=self.output)
        # self.store('vis')
        pass

    def enable_fp16_training(self, with_criterion=False):
        if not self.cfg.fp16:
            return
        self.logger.info(f'==> Enable fp16 training')
        if self.amp_scaler is not None:
            self.store('amp_scaler')
        # if self.amp is None:
        #     return
        with_criterion = with_criterion and isinstance(self.criterion, nn.Module)
        # self.amp.register_float_function(ext.layers.dcn_v2, 'dcn_v2_conv')
        # if with_criterion:
        #     (self.model, self.criterion), self.optimizer = self.amp.initialize([self.model, self.criterion],
        #                                                                        self.optimizer, **self.cfg.amp_cfg)
        # else:
        #     self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, **self.cfg.amp_cfg)
        if not self.cfg.distributed or self.cfg.test:
            return
        self.model = self._to_parallel(self.model)
        if with_criterion:
            self.criterion = self._to_parallel(self.criterion)

    def enable_train_time_estimate(self):
        self.hook_manager.add_hook(self.train_timer.start, 'before_train_epoch')
        self.hook_manager.add_hook(self.train_timer.step, 'after_train_step')

    def execute_context(self):
        return ext.utils.net_no_sync(
            self.model, not ext.utils.check_interval(self.global_step + 1, self.interval_grad_acc)
        )

    def execute_backward(self, loss, optimizer=None, between_fun=None):
        # type: (Union[Tensor, Dict[str, Tensor]],  Optional[Optimizer], Optional[Callable]) -> Tensor
        self.global_step += 1
        if not isinstance(loss, Tensor):
            loss = sum(loss.values())
        if optimizer is None:
            optimizer = self.optimizer
        if self.step == 0:
            # self._num_accumulated_steps = 0
            optimizer.zero_grad(set_to_none=True)
        ## skip error gradient
        if (not self.cfg.fp16) and (torch.isnan(loss) or torch.isinf(loss)):
            self.logger.error(f'==> Loss is {loss}')
            # loss.backward()
            exit(1)
            # return loss
        ## backward
        if self.interval_grad_acc != 1:
            loss = loss * (1. / self.interval_grad_acc)
        if self.cfg.fp16:
            self.amp_scaler.scale(loss).backward()
        else:
            loss.backward()
        # self._num_accumulated_steps += 1
        if between_fun is not None:
            between_fun()
        ## update parameters
        if self.step % self.interval_grad_acc != 0:
            return loss
        if self.cfg.grad_clip > 0:
            if self.cfg.fp16:
                self.amp_scaler.unscale_(optimizer)
                parameters = self.model.parameters()
            else:
                parameters = self.model.parameters()
            torch.nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip)
        if self.cfg.fp16 and self.amp_scaler is not None:
            self.amp_scaler.step(optimizer)
            self.amp_scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True here can modestly improve performance
        return loss

    def forward_and_backward(self, inputs: tuple, targets=tuple(), input_dict: dict = None, target_dict: dict = None):
        input_dict = {} if input_dict is None else input_dict
        target_dict = {} if target_dict is None else target_dict
        inputs, targets = ext.utils.tensor_to(inputs, targets, device=self.device, non_blocking=True)
        with self.execute_context():
            with ext.ops.autocast(enabled=self.cfg.fp16):
                outputs = self.model(*inputs, **input_dict)
                losses_dict = self.criterion(outputs, *targets, **target_dict)  # type: Dict[str, torch.Tensor]
                # losses_dict = self.criterion(*outputs, *targets, **target_dict)  # type: Dict[str, torch.Tensor]
                losses = sum(losses_dict.values())  # type: torch.Tensor # noqa
            self.execute_backward(losses)
        return losses, losses_dict

    def to_cuda(self, find_unused_parameters=False, names: Sequence[str] = None):
        assert self.model is not None, "Please define model first"
        ext.optimizer.freeze_modules(self.cfg, self.model)  # 冻结特定的层
        self.model.to(self.device)
        if self.criterion and isinstance(self.criterion, nn.Module):  # Some criterion is in the model
            self.criterion.to(self.device)
        if not self.cfg.distributed or self.cfg.test:
            return
        # if self.cfg.fp16 and self.amp is not None:
        #     return
        if names is None:
            self.model = self._to_parallel(self.model, find_unused_parameters=find_unused_parameters)
        else:
            for name in names:
                self.logger.info(f'submodel: {name}')
                assert hasattr(self.model, name), f"{name} is not the part of model"
                setattr(self.model, name, self._to_parallel(getattr(self.model, name), find_unused_parameters))

    def _to_parallel(self, model, find_unused_parameters=False):
        if self.cfg.dist_apex and apex is not None:
            if self.cfg.sync_bn:
                model = apex.parallel.convert_syncbn_model(model)
                self.logger.info("==> convert to SyncBatchNorm.")
            model = apex.parallel.DistributedDataParallel(model)
            self.logger.info("==> use DistributedDataParallel by apex")
        else:
            if self.cfg.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.logger.info("==> convert to SyncBatchNorm.")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.cfg.gpu_id],
                output_device=self.cfg.gpu_id,
                find_unused_parameters=find_unused_parameters
            )
            self.logger.info("==> use DistributedDataParallel by torch")
        return model

    @property
    def mode(self):
        if self.cfg.test:
            return "test"
        if self.cfg.eval:
            return "eval"
        return "train"

    @property
    def training(self):
        return not (self.cfg.test or self.cfg.eval)

    def save_model(self, name="model.pth", net=None):
        if not ext.is_main_process():
            return
        path = os.path.join(self.output, name)
        data = (self.model if net is None else net).state_dict()
        data = ext.utils.state_dict_strip_prefix_if_present(data, "module.")
        self.logger.info(f"Saving model to {path}")
        if torch.__version__ >= '1.6.0':
            torch.save(data, path, _use_new_zipfile_serialization=True)
        else:
            torch.save(data, path)
        return

    def load_model(self):
        if not self.cfg.load or self.cfg.resume:
            return
        self.logger.info('==> Loading model from {}, strict: {}'.format(self.cfg.load, not self.cfg.load_no_strict))
        loaded_state_dict = torch.load(self.cfg.load, map_location=torch.device("cpu"))
        loaded_state_dict = ext.utils.convert_pth(loaded_state_dict, **self.cfg.load_cfg)
        # for DataParallel or DistributedDataParallel
        # loaded_state_dict = ext.utils.state_dict_strip_prefix_if_present(loaded_state_dict, "module.")
        return self.model.load_state_dict(loaded_state_dict, strict=not self.cfg.load_no_strict)

    def store(self, name: str, attr: str = None):
        """
        Store and resume the attribution <name>
        if <name> can "load_state_dict" and "state_dict", using them. (For model, optimizer, lr_scheduler)
        """
        self.checkpoint_manager.store(name, self, attr)

    def resume(self, name, default=None):
        """
        Load <name> item form checkpoint
        """
        value = self.checkpoint_manager.resume(name)
        if value is None:
            return default
        self.logger.info("==> Resume `{}`={}".format(name, value))
        return value

    def set_output_dir(self, *path, log_filename=None):
        if self.cfg.output_dir is not None:
            self.output = Path(self.cfg.output_dir)
        else:
            if len(path) == 0:
                if hasattr(self.cfg, 'exp_name'):
                    path = [self.cfg.exp_name]
                else:
                    path = [f"{self.model_name}_{self.model_cfg}".strip('_')]
            self.output = Path(self.cfg.output).joinpath(*path, str(self.cfg.log_suffix))
        self.output = self.output.expanduser()
        self.output.mkdir(exist_ok=True, parents=True)
        self.checkpoint_manager.set_save_dir(self.output)

    def execute_hooks(self, hook_type='before_train_epoch'):
        getattr(self.hook_manager, hook_type)()

    def add_hooks(self, f: Callable, hook_type='before_train_epoch', *args, **kwargs):
        self.hook_manager.add_hook(f, hook_type, *args, **kwargs)

    def enable_model_hooks(self, *modules: nn.Module):
        num_add_hook = self.hook_manager.add_module_hooks(*modules)
        if num_add_hook > 0:
            self.logger.info(f'==> Add {num_add_hook} module hooks')

    def _set_now_state(self, step=None, epoch=None, is_during_training=None):
        if step is not None:
            self.step = step
        if epoch is not None:
            self.epoch = epoch
        if is_during_training is not None:
            self.is_during_training = is_during_training

    def run(self):
        pass


# 常规框架,每遍历一次data_loader为一个epoch, 共运行self.cfg.epochs次train_epoch
class Framework(BaseFramework):

    def __init__(
        self,
        config_args=None,
        m_data_transforms=None,
        m_datasets=None,
        m_data_loader=None,
        *args,
        **kwargs
    ):
        super().__init__(config_args, m_data_transforms, m_datasets, m_data_loader, *args, **kwargs)
        self.hook_manager.add_hook(lambda: self._set_now_state(step=0, is_during_training=True), 'before_train_epoch')
        self.hook_manager.add_hook(lambda: self._set_now_state(step=self.step + 1), 'after_train_step')
        self.hook_manager.add_hook(lambda: self._set_now_state(epoch=self.epoch + 1), 'after_train_epoch')
        self.hook_manager.add_hook(lambda: self._set_now_state(step=0, is_during_training=False), 'before_eval_epoch')
        self.hook_manager.add_hook(lambda: self._set_now_state(step=self.step + 1), 'after_eval_step')
        self.configure(config_args, *args, **kwargs)

    def step_2_environment(self, *args, **kwargs):
        super().step_2_environment(*args, **kwargs)
        self.store('epoch')
        self.num_epochs = self.cfg.epochs

    def step_5_data_loader_and_transform(self):
        if self.train_db is not None:
            if self._m_data_transforms is not None:
                self.train_db.set_transforms(self._m_data_transforms.make(self.cfg, mode='train'))  # noqa
            if self._m_data_loader is not None:
                self.train_loader = self._m_data_loader.make(self.cfg, self.train_db, mode='train')
                self.num_steps = len(self.train_loader)
            self.logger.info('==> Train db:', self.train_db)

        if self.eval_db is not None:
            if self._m_data_transforms is not None:
                self.eval_db.set_transforms(self._m_data_transforms.make(self.cfg, mode='eval'))  # noqa
            if self._m_data_loader is not None:
                self.eval_loader = self._m_data_loader.make(self.cfg, self.eval_db, mode='eval')
            self.logger.info('==> Eval db:', self.eval_db)

        if self.test_db is not None:
            if self._m_data_transforms is not None:
                self.test_db.set_transforms(self._m_data_transforms.make(self.cfg, mode='test'))  # noqa
            if self._m_data_loader is not None:
                self.test_loader = self._m_data_loader.make(self.cfg, self.test_db, mode='test')
            self.logger.info('==> Test db:', self.test_db)

    def step_7_lr(self, *args, **kwargs):
        if self.mode != 'train':
            return
        self.lr_scheduler = ext.lr_scheduler.make(self.optimizer, self.cfg, self.num_steps)
        self.store("lr_scheduler")
        self.hook_manager.add_hook(
            lambda: self.lr_scheduler.step(step=self.step, epoch=self.epoch), 'before_train_step'
        )
        return

    def save_checkpoint(self, filename="checkpoint.pth", **kwargs):
        self.checkpoint_manager.save(filename, epoch=self.epoch, save_dir=self.output, **kwargs)

    def run(self):
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger.info("{} Begin {} at {}".format('=' * 20, self.mode, now_date.replace('_', ' '), '=' * 20))
        if self.cfg.test:
            self.test()
        elif self.cfg.eval:
            self.hook_manager.before_eval_epoch()
            self.evaluation('eval')
            self.hook_manager.after_eval_epoch()
        else:
            self.hook_manager.before_train()
            self.train_timer.reset((self.num_epochs - self.epoch) * self.num_steps)
            for self.epoch in range(self.epoch, self.cfg.epochs):
                self.hook_manager.before_train_epoch()
                self.model.train()
                self.train_epoch()
                self.hook_manager.after_eval_epoch()
                self.save_checkpoint()
                if ext.utils.check_interval(self.epoch, self.cfg.eval_interval, self.num_epochs):
                    self.hook_manager.before_eval_epoch()
                    self.evaluation()
                    self.hook_manager.after_eval_epoch()
                self.logger.info('')
                if self.cfg.debug:
                    exit(1)
            self.save_model('last.pth')
            self.hook_manager.after_train()
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger.info("======================  End {} ========================".format(now_date.replace('_', ' ')))
        if not ext.is_main_process():
            return
        if self.metric_manager is None or self.metric_manager.best_score is None:
            ext.my_logger.copy_to(suffix=f'_{now_date}', ext='.txt')
        else:
            score = f"score={ext.utils.float2str(self.metric_manager.best_score, 8)}"
            ext.my_logger.copy_to(suffix=f'_{now_date}_{score}', ext='.txt')
        if self.training:
            if self.metric_manager is None or self.metric_manager.best_score is None:
                self.save_model(f'model_{now_date}.pth')
            elif self.output.joinpath('best.pth').exists():
                score = f"score={ext.utils.float2str(self.metric_manager.best_score, 8)}"
                shutil.copyfile(self.output.joinpath('best.pth'), self.output.joinpath(f'{score}_{now_date}.pth'))
            self.checkpoint_manager.remove_all()
            ext.config.save(self.cfg, os.path.join(self.output, 'config.yaml'))

    def train_epoch(self):
        pass

    def evaluation(self, name=''):
        # if self.metric_manager.is_best:
        #     self.save_model('best.pth')
        pass

    def test(self):
        pass


# 总共训练iterations次
class IterableFramework(BaseFramework):

    def __init__(
        self,
        config_args=None,
        m_data_transforms=None,
        m_datasets=None,
        m_data_loader=None,
        *args,
        **kwargs
    ):
        super().__init__(config_args, m_data_transforms, m_datasets, m_data_loader, *args, **kwargs)
        self.hook_manager.add_hook(
            lambda: self._set_now_state(step=self.step, is_during_training=True), 'before_train_epoch',
            insert=0,
        )
        self.hook_manager.add_hook(lambda: self._set_now_state(step=self.step + 1), 'after_train_step', insert=0)
        self.hook_manager.add_hook(lambda: self._set_now_state(epoch=0), 'after_train_epoch', insert=0)
        self.hook_manager.add_hook(lambda: self._set_now_state(is_during_training=False),
                                   'before_eval_epoch',
                                   insert=0)
        self.configure(config_args, *args, **kwargs)

    def step_2_environment(self, *args, **kwargs):
        super().step_2_environment(*args, **kwargs)
        self.store('step')
        if self.cfg.start_epoch is not None:
            self.step = self.cfg.start_epoch
        self.num_steps = self.cfg.epochs

    def step_5_data_loader_and_transform(self):
        if self.train_db is not None:
            if self._m_data_transforms is not None:
                self.train_db.set_transforms(self._m_data_transforms.make(self.cfg, mode='train'))  # noqa
            if self._m_data_loader is not None:
                self.train_loader = self._m_data_loader.make(
                    self.cfg, self.train_db, mode='train', batch_sampler='iterable'
                )
            self.logger.info(f'==> Train db: {self.train_db}')

        if self.eval_db is not None:
            if self._m_data_transforms is not None:
                self.eval_db.set_transforms(self._m_data_transforms.make(self.cfg, mode='eval'))  # noqa
            if self._m_data_loader is not None:
                self.eval_loader = self._m_data_loader.make(self.cfg, self.eval_db, mode='eval')
            self.logger.info(f'==> Eval db: {self.eval_db}')

        if self.test_db is not None:
            if self._m_data_transforms is not None:
                self.test_db.set_transforms(self._m_data_transforms.make(self.cfg, mode='test'))  # noqa
            if self._m_data_loader is not None:
                self.test_loader = self._m_data_loader.make(self.cfg, self.test_db, mode='test')
            self.logger.info(f'==> Test db: {self.test_db}')

    def step_7_lr(self, *args, **kwargs):
        if self.mode != 'train':
            return
        self.lr_scheduler = ext.lr_scheduler.make(self.optimizer, self.cfg, self.num_steps)
        self.store("lr_scheduler")
        self.hook_manager.add_hook(lambda: self.lr_scheduler.step(step=0, epoch=self.step), 'before_train_step')
        return

    def save_checkpoint(self, filename="checkpoint.pth", **kwargs):
        kwargs.setdefault('epoch', self.step)
        kwargs.setdefault('save_dir', self.output)
        self.checkpoint_manager.save(filename, **kwargs)

    def run(self):
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger.info('{} Begin {} at {} {}'.format('=' * 20, self.mode, now_date.replace('_', ' '), '=' * 20))
        if self.cfg.test:
            self.test()
        elif self.cfg.eval:
            # self.hook_manager.before_eval_epoch()
            self.evaluation('eval')
            # self.hook_manager.after_eval_epoch()
        else:
            self.num_epochs = 1
            self.epoch = 0
            self.train_timer.reset(self.num_steps - self.step)
            data_iter = iter(self.train_loader)

            self.model.train()
            self.hook_manager.before_train()
            self.hook_manager.before_train_epoch()
            for self.step in range(self.step, self.num_steps):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    data = next(data_iter)
                # self.execute_hooks('before_train_step')
                self.train_step(data)
                # self.execute_hooks('after_train_step')
                self.save_checkpoint()
                if ext.utils.check_interval(self.step, self.cfg.eval_interval, self.num_steps):
                    self.hook_manager.after_train_epoch()
                    # self.hook_mananger.before_eval_epoch()
                    self.evaluation()
                    # self.hook_mananger.after_eval_epoch()
                    self.model.train()
                    self.hook_manager.before_train_epoch()
                    self.logger.info('')
                if self.cfg.debug:
                    exit(1)
            self.hook_manager.after_train_epoch()
            self.hook_manager.after_train()
            self.save_model('last.pth')
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger.info("======================  End {} ========================".format(now_date.replace('_', ' ')))
        if not ext.is_main_process():
            return
        if self.metric_manager is None or self.metric_manager.best_score is None:
            ext.my_logger.copy_to(suffix=f"_{now_date}", ext='.txt')
        else:
            score = f"score={ext.utils.float2str(self.metric_manager.best_score, 8)}"
            ext.my_logger.copy_to(suffix=f"_{now_date}_{score}", ext='.txt')
        if self.training:
            if self.metric_manager is None or self.metric_manager.best_score is None:
                self.save_model(f'model_{now_date}.pth')
            elif self.output.joinpath('best.pth').exists():
                score = f"score={ext.utils.float2str(self.metric_manager.best_score, 8)}"
                shutil.copyfile(self.output.joinpath('best.pth'), self.output.joinpath(f'{score}_{now_date}.pth'))
            self.checkpoint_manager.remove_all()
            ext.config.save(self.cfg, os.path.join(self.output, 'config.yaml'))

    def train_step(self, data):
        pass

    def evaluation(self, name=''):
        pass

    def test(self):
        pass
