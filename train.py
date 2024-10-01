import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import my_ext as ext
from my_ext import utils, ops_3d
import networks, data_loader, datasets
# from datasets.NerfiesDataset import NerfiesDataset
from datasets.colmap_dataset import ColmapDataset, fetchPly, storePly
# from datasets.colmap_dynerf_dataset import DyNeRFColmapDataset
from networks.gaussian_splatting import BasicPointCloud, SH2RGB


class GaussianTrainTask(ext.IterableFramework):
    model: networks.gaussian_splatting.GaussianSplatting
    train_db: Optional[datasets.NERF_Base_Dataset]
    eval_db: Optional[datasets.NERF_Base_Dataset]
    test_db: Optional[datasets.NERF_Base_Dataset]

    def __init__(self, *args, **kwargs):
        super(GaussianTrainTask, self).__init__(*args, m_data_loader=data_loader, m_datasets=datasets, **kwargs)

    def extra_config(self, parser):
        networks.build.options(parser)
        parser.add_argument('--exp-name', default='nerf', help='The name of experiments')
        utils.add_cfg_option(parser, '--train-kwargs', help='extra train kwargs')
        utils.add_cfg_option(parser, '--eval-kwargs', help='extra eval kwargs')
        utils.add_cfg_option(parser, '--test-kwargs', help='extra test kwargs')
        utils.add_bool_option(parser, '--save-video', default=True, help='Save the test results to vedio')
        parser.add_argument('--eval-num-steps', default=-1, type=int, help='The steps when evaluate during training')
        parser.add_argument('--vis-interval', default=1_000, type=int)
        utils.add_cfg_option(parser, '--vis-kwargs', help='The config for visualize')
        utils.add_bool_option(parser, '--vis-clear', default=None, help='clear visualization')
        parser.add_argument('--num-init-points', default=100_000, type=int)
        utils.add_path_option(parser, '--init-ply', default=None)
        utils.add_bool_option(parser, '--random-pcd', default=False)
        utils.add_bool_option(parser, '--lr-scheduler-in-model', default=True)
        super().extra_config(parser)

    def step_2_environment(self, *args, **kwargs):
        super().step_2_environment(output_paths=(self.cfg.exp_name, self.cfg.scene))
        if self.cfg.weighted_sample:
            self.cfg.num_workers = 0

    def step_3_dataset(self, *args, **kwargs):
        super().step_3_dataset(*args, **kwargs)

        if not self.cfg.random_pcd and self.cfg.init_ply is not None:
            self.logger.info(f"try to Load init point cloud from: {self.cfg.init_ply}")
            pcd = fetchPly(self.cfg.init_ply)
        # elif not self.cfg.random_pcd and isinstance(self.train_db, (ColmapDataset, DyNeRFColmapDataset)):
        #     pcd = self.train_db.point_cloud
        #     logging.info(f"[red]load point cloud from colmap")
        # elif not self.cfg.random_pcd and isinstance(self.train_db, NerfiesDataset) and self.train_db.points is not None:
        #     xyz = self.train_db.points
        #     shs = np.random.random((len(xyz), 3)) / 255.0
        #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))
        else:
            # ply_path = self.train_db.root.joinpath("points3d.ply")
            # if 1 or not ply_path.exists():
            # Since this data set has no colmap data, we start with random points
            num_pts = self.cfg.num_init_points

            self.logger.info(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            if hasattr(self.train_db, 'scene_size'):
                min_v = self.train_db.scene_center - self.train_db.scene_size * 0.5
                xyz = np.random.random((num_pts, 3)) * self.train_db.scene_size + min_v  # noqa
            else:
                xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
                self.logger.warning("scene bound are set to [-1.3, 1.3]")
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            # storePly(ply_path, xyz, SH2RGB(shs) * 255)

            # try:
            #     pcd = fetchPly(ply_path)
            # except:
            #     pcd = None
        self._pcd = pcd

    def step_4_model(self, *args, **kwargs):
        self.set_output_dir(self.cfg.exp_name, self.cfg.scene)
        self.model = networks.build.make(self.cfg)  # noqa
        self.model.set_from_dataset(utils.fnn(self.train_db, self.eval_db, self.test_db))
        self.criterion = self.model.loss

        self.load_model()
        self.store('model')
        if not self.cfg.load and not self.cfg.resume:
            self.model.create_from_pcd(self._pcd)
            self.logger.info('create_from_pcd')
            storePly(
                self.output.joinpath('init_points.ply'),
                self.model.points.detach().cpu().numpy(),
                SH2RGB(self.model._features_dc[:, 0].detach().cpu().numpy()) * 255
            )
        self.model.training_setup()
        # if self.mode != 'train':
        #     self.model.active_sh_degree = self.model.max_sh_degree
        self.to_cuda()
        # torch.set_anomaly_enabled(True)
        self.logger.info(f"==> Model: {self.model}")
        self.model._task = self

    def step_5_data_loader_and_transform(self):
        if self.train_db is not None:
            if self._m_data_loader is not None:
                self.train_loader = self._m_data_loader.make(
                    self.cfg, self.train_db, mode='train', batch_sampler='iterable')
            self.logger.info(f'==> Train db: {self.train_db}')

        if self.eval_db is not None:
            if self._m_data_loader is not None:
                self.eval_loader = self._m_data_loader.make(self.cfg, self.eval_db, mode='eval', batch_size=1)
            self.logger.info(f'==> Eval db: {self.eval_db}')

        if self.test_db is not None:
            if self._m_data_loader is not None:
                self.test_loader = self._m_data_loader.make(self.cfg, self.test_db, mode='test', batch_size=1)
            self.logger.info(f'==> Test db: {self.test_db}')

    def step_6_optimizer(self, *args, **kwargs):
        if self.mode != 'train':
            return
        m = utils.get_net(self.model)
        if hasattr(m, 'get_params'):
            self.optimizer = ext.optimizer.make(None, self.cfg, m.get_params(self.cfg))
        else:
            self.optimizer = ext.optimizer.make(self.model, self.cfg)
        self.store("optimizer")
        return

    def step_7_lr(self, *args, **kwargs):
        if hasattr(self.model, 'update_learning_rate') and self.cfg.lr_scheduler_in_model:
            self.hook_manager.add_hook(self.model.update_learning_rate, 'before_train_step')
        else:
            super().step_7_lr(*args, **kwargs)

    def step_8_others(self, *args, **kwargs):
        super().step_8_others(*args, **kwargs)
        self.hook_manager.add_hook(
            lambda: ext.trainer.
            change_with_training_progress(self.model, self.step, self.num_steps, self.epoch, self.num_epochs),
            'before_train_step'
        )
        self.hook_manager.add_hook(
            lambda: self.logger.info(f"Peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.3f} GiB"),
            'after_train'
        )
        if self.cfg.vis_clear is not None:
            clear_vis = self.cfg.vis_clear
        else:
            clear_vis = self.mode == 'train' and (not self.cfg.debug and not self.cfg.resume)
        if clear_vis:
            utils.dir_create_empty(self.output.joinpath('vis'))
        else:
            self.output.joinpath('vis').mkdir(exist_ok=True, parents=True)
        # self.hook_manager.add_hook(self.visualize, 'after_train_step')

    def run(self):
        self.loss_dict_meter = ext.DictMeter(float2str=utils.float2str)
        self.losses_meter = ext.AverageMeter()
        self.psnr_meter = ext.AverageMeter()

        self.progress = ext.utils.Progress(enable=ext.is_main_process())
        if self.mode == 'train':
            self.progress.add_task('train', self.num_steps, self.step)
        if self.mode != 'test' and self.eval_loader is not None:
            self.progress.add_task('eval', len(self.eval_loader))
        with self.progress:
            super().run()

    def train_step(self, data):
        inputs, targets, infos = ext.utils.tensor_to(data, device=self.device, non_blocking=True)
        self.progress.start('train')
        self.model.train()
        self.hook_manager.before_train_step()
        if self.cfg.debug:
            self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
            self.logger.debug(f'targets: {utils.show_shape(targets)}')
            self.logger.debug(f'infos: {utils.show_shape(infos)}')
        with self.execute_context():
            with ext.autocast(self.cfg.fp16):
                if hasattr(self.model, 'render'):
                    outputs = self.model.render(**inputs, **self.cfg.train_kwargs, info=infos)
                else:
                    outputs = self.model(**inputs, **self.cfg.train_kwargs, info=infos)
                if self.cfg.debug:
                    self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
            loss_dict = self.criterion(inputs, outputs, targets, infos)
            if self.cfg.debug:
                self.logger.debug(f'loss_dict: {loss_dict}')
            # if self.cfg.add_noise_interval[0] > 0:
            #     error_map = (outputs['images'] - targets['images'][..., :, :, :3]).norm(dim=-1)
            #     error_map = F.avg_pool2d(error_map[:, None, :, :], 7, 2, 3)[0, 0]
            #     yx = error_map.argmax()
            #     x, y = (yx % error_map.shape[-1]), yx // error_map.shape[-1]
            #     # plt.imshow((error_map / error_map.max()).detach().cpu().numpy())
            #     # plt.scatter(x.item(), y.item())
            #     # plt.show()
            #     x, y = (x / error_map.shape[-1] * infos['size'][0]), (y / error_map.shape[-2] * infos['size'][1])
            #     # print(x, y)
            # else:
            #     error_map = None
            losses = self.execute_backward(
                loss_dict, between_fun=lambda: self.model.adaptive_control(inputs, outputs, self.optimizer, self.step)
            )
            if utils.check_interval(self.step + 1, self.cfg.vis_interval, self.cfg.epochs):
                gt = targets['images'][..., :3]
                gt = gt[0] if gt.ndim == 4 else gt
                image = outputs['images']
                image = image[0] if image.ndim == 4 else image
                diff = (image - gt).abs()
                image = torch.cat([image, gt, diff], dim=1)
                utils.save_image(self.output.joinpath('vis', f'train_{self.step + 1}.png'), image)
                del gt, diff, image
        self.loss_dict_meter.update(loss_dict)
        self.losses_meter.update(losses)

        with torch.no_grad():
            if 'mse' in loss_dict:
                mse = loss_dict['mse']
            else:
                mse = F.mse_loss(outputs['images'][..., :3].reshape(-1), targets['images'][..., :3].reshape(-1))
            psnr = -10 * torch.log10(mse)
            # if self.cfg.weighted_sample:
            #     self.train_db.update_errors(outputs['images'], targets['images'])
        self.psnr_meter.update(psnr)
        self.hook_manager.after_train_step()

        if ext.utils.check_interval(self.step, self.cfg.print_f, self.num_steps):
            lr = [g['lr'] for g in self.optimizer.param_groups if g.get('name', None) == 'xyz'][0]
            self.logger.info(
                f"[{self.step}]/[{self.num_steps}]: "
                f"loss={utils.float2str(self.losses_meter.avg)}, {self.loss_dict_meter.average}, "
                f"psnr={utils.float2str(self.psnr_meter.avg)}, "
                f"lr={utils.float2str(lr)}, "
                f"{self.train_timer.progress}",
            )
            self.psnr_meter.reset()
            self.loss_dict_meter.reset()
            self.losses_meter.reset()
        self.progress.step('train')
        self.visualize()

    def eval_step(self, step, data, **eval_kwargs):
        self.hook_manager.before_eval_step()
        # self.logger.debug(f'inputs: {utils.show_shape(data[0])}')
        # self.logger.debug(f'targets: {utils.show_shape(data[1])}')
        # self.logger.debug(f'infos: {utils.show_shape(data[2])}')

        inputs, targets, infos = utils.tensor_to(*data, device=self.device, non_blocking=True)
        # inputs = {k: None if v is None else v.squeeze(0) for k, v in inputs.items()}
        # targets = {k: None if v is None else v.squeeze(0) for k, v in targets.items()}
        self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
        self.logger.debug(f'targets: {utils.show_shape(targets)}')
        self.logger.debug(f'infos: {utils.show_shape(infos)}')
        # self.logger.debug(f"split: {utils.show_shape(inputs, targets, infos)}")
        with ext.autocast(self.cfg.fp16):
            if hasattr(self.model, 'render'):
                outputs = self.model.render(**inputs, **eval_kwargs, info=infos)
            else:
                outputs = self.model(**inputs, **eval_kwargs, info=infos)
        pred_images = outputs['images'].clamp(0., 1.)
        self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
        loss_dict = self.criterion(inputs, outputs, targets, infos)
        self.logger.debug(f'losses: {loss_dict}')
        self.metric_manager.update('image', pred_images[..., :3], data[1]['images'][..., :3])
        self.metric_manager.update('loss', sum(loss_dict.values()), **loss_dict)
        if self.cfg.debug:
            plt.figure(dpi=200)
            plt.subplot(121)
            plt.imshow(utils.as_np_image(data[1]['images'].flatten(0, -4)[0]))
            plt.title('gt')
            plt.subplot(122)
            plt.imshow(utils.as_np_image(pred_images.flatten(0, -4)[0]))
            plt.title('predict')
            plt.show()

        if step == 0:
            gt = targets['images'][..., :3]
            gt = gt[0] if gt.ndim == 4 else gt
            diff = (pred_images[0] - gt).abs()
            image = torch.cat([pred_images[0], gt, diff], dim=1)
            utils.save_image(self.output.joinpath('vis', f'eval_{self.step + 1}.png'), image)
            del gt, diff, image

        self.progress.step('eval', self.metric_manager.str())
        self.hook_manager.after_eval_step()
        return

    def evaluation(self, name=''):
        if self.mode == 'train':
            self.progress.pause('train')
        self.hook_manager.before_eval_epoch()
        self.model.eval()
        self.progress.reset('eval', start=True)
        self.progress.start('eval', len(self.eval_loader))
        eval_kwargs = self.cfg.eval_kwargs.copy()
        batch_size = eval_kwargs.pop('batch_size', self.cfg.batch_size[1])
        for step, data in enumerate(self.eval_loader):
            self.eval_step(step, data, **eval_kwargs)
            if self.mode == 'train' and 0 < self.cfg.eval_num_steps <= step:
                break
            if self.cfg.debug:
                break
        self.hook_manager.after_eval_epoch()
        self.logger.info(f"Eval [{self.step}/{self.num_steps}]: {self.metric_manager.average}")
        if self.mode == 'train':
            if self.metric_manager.is_best:
                self.save_model('best.pth')
            self.progress.stop('eval')
        return

    @torch.no_grad()
    def visualize(self, index=None):
        if not utils.check_interval(self.step, self.cfg.vis_interval):
            return
        self.model.eval()
        torch.cuda.empty_cache()
        vis_kwargs = self.cfg.vis_kwargs.copy()  # type: dict
        batch_size = self.cfg.batch_size[1]
        self.progress.pause('train')
        if index is None:
            index = np.random.randint(0, len(self.train_db))
        logging.info(f"visualize image {index} as step {self.step}")
        inputs, targets, info = utils.tensor_to(*self.train_db[index], device=self.device, non_blocking=True)
        inputs = {k: utils.to_tensor(v, device=self.device) for k, v in inputs.items()}
        targets = {k: utils.to_tensor(v, device=self.device) for k, v in targets.items()}
        self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
        self.logger.debug(f'targets: {utils.show_shape(targets)}')
        self.logger.debug(f'info: {utils.show_shape(info)}')
        info = utils.tensor_to(info, device=self.device)
        if hasattr(self.model, 'render'):
            outputs = self.model.render(**inputs, **vis_kwargs, info=info)
        else:
            outputs = self.model(**inputs, **vis_kwargs, info=info)
        self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
        images = outputs['images'] if 'images' in outputs else None
        images_c = outputs['images_c'] if 'images_c' in outputs else None

        cat_dim = 0 if self.train_db.aspect > 1. else 1
        if images[0] is not None:
            img_pred = images[..., :3].cpu()
            img_gt = targets['images'][..., :3].cpu()
            if img_pred.ndim == 4:
                assert img_pred.shape[0] == 1
                img_pred = img_pred[0]
            image_list = [img_pred, img_gt, (img_pred - img_gt).abs()]
            if images_c is not None:
                image_list.append(images_c[0, ..., :3].cpu())
            image = torch.cat(image_list, dim=cat_dim)
            utils.save_image(self.output.joinpath('vis', f"img_{self.step}_{index}.png"), image)
        self.model.train()

    def load_model(self):
        if not self.cfg.load or self.cfg.resume:
            return
        if self.cfg.load.suffix == '.ply':
            self.model.load_ply(self.cfg.load)
            logging.warning(f"Load ply from {self.cfg.load}")
        else:
            super().load_model()

    def save_model(self, name="point_cloud.ply", net=None):
        if not ext.is_main_process():
            return
        if name.endswith('.ply'):
            self.model.save_ply(self.output.joinpath(name).with_suffix('.ply'))
            self.logger.info(f"save model to {self.output.joinpath(name).with_suffix('.ply')}")
        else:
            super().save_model(name, net)


if __name__ == '__main__':
    GaussianTrainTask().run()
# speed test: 1000 steps
#         data,  forward, loss, backward, optimizer, other, total
# my sum,  3s951,  3s325, 3s726, 8s001,   3s872,     905.ms, 23s780
# origin, 27.0ms,  2s868, 2s903, 7s204,   2s231,     128.ms, 15s361
# diff  ,  3s924,  457ms, 823ms, 0.797,   1.641,     777.ms,  8s419
# after train step=115.ms, progress=94.3ms, data= 3s951, before train_step=220.ms,
# forward= 3s325, loss= 3s726, backward= 8s001, optimize= 3s872, meter=476.ms
