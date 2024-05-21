import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import datasets as datasets
import networks as networks
import my_ext as ext
from my_ext import ops_3d, utils


# noinspection PyAttributeOutsideInit
class NeRFTask(ext.IterableFramework):
    model: networks.NeRF_Network

    def __init__(self, *args, **kwargs):
        super().__init__(*args, m_datasets=datasets, **kwargs)

    def extra_config(self, parser):
        networks.options(parser)
        parser.add_argument('--exp-name', default='nerf', help='The name of experiments')
        utils.add_cfg_option(parser, '--train-kwargs', help='extra train kwargs')
        utils.add_cfg_option(parser, '--eval-kwargs', help='extra eval kwargs')
        utils.add_cfg_option(parser, '--test-kwargs', help='extra test kwargs')
        utils.add_bool_option(parser, '--save-video', default=True, help='Save the test results to vedio')
        parser.add_argument('--eval-num-steps', default=-1, type=int, help='The steps when evaluate during training')
        parser.add_argument('--lr-geo', default=1., type=float, help='The learning rate rate of geometry in nvidiffrec')
        utils.add_bool_option(parser, '--optim-camera-pose', default=False)
        # utils.add_choose_option(parser, 'scene_type', choices=['synthetic', 'forwardfacing', 'real360'])
        # utils.add_choose_option(parser, '--color-space', choices=['linear', 'srgb'])
        # utils.add_bool_option(parser, '--gui', default=False)
        parser.add_argument('--mesh', default='', help='Generate mesh when given name')
        parser.add_argument('-R', '--resolution', default=64, type=int, help='march cubes')
        parser.add_argument('--eval-mesh-interval', default=0, type=int,
            help='Generate mesh for evalution given interval')
        utils.add_bool_option(parser, '--mesh-scale', default=False, help='apply scale on mesh')

        parser.add_argument('--vis-interval', default=1_000, type=int)
        utils.add_cfg_option(parser, '--vis-kwargs', help='The config for visualize')
        utils.add_bool_option(parser, '--vis-clear', default=None, help='clear visualization')

    def step_2_environment(self, *args, **kwargs):
        super().step_2_environment(output_paths=(self.cfg.exp_name, self.cfg.scene))
        if self.cfg.weighted_sample:
            self.cfg.num_workers = 0

    def step_3_dataset(self, *args, **kwargs):
        if self._m_datasets is None:
            return
        self.train_db = self._m_datasets.make(self.cfg, mode='train')  # type: datasets.NERF_Base_Dataset
        if self.mode == 'eval' or (self.mode == 'train' and self.cfg.eval_interval > 0):
            self.eval_db = self._m_datasets.make(self.cfg, mode='eval')  # type: datasets.NERF_Base_Dataset
        if self.mode == 'test':
            self.test_db = self._m_datasets.make(self.cfg, mode='test')  # type: datasets.NERF_Base_Dataset
        return

    def step_4_model(self, *args, **kwargs):
        self.set_output_dir(self.cfg.exp_name, self.cfg.scene)
        if self.cfg.optim_camera_pose:
            self.model = networks.make(self.cfg, gt_poses=self.train_db.Tv2w, images=self.train_db.images)
        else:
            self.model = networks.make(self.cfg)
        self.model.set_from_dataset(utils.fnn(self.train_db, self.eval_db, self.test_db))
        self.criterion = self.model.loss

        # if isinstance(self.model, networks.instance_ngp.Instant_NGP_Renderer):
        #     self.add_hooks(self.model.mark_untrained_grid, 'before_train', self.train_db.Tv2w, self.train_db.Tv2s)

        self.load_model()
        self.store('model')
        self.to_cuda()
        # torch.set_anomaly_enabled(True)
        self.logger.info(f"==> Model: {self.model}")

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

    def step_8_others(self, *args, **kwargs):
        super().step_8_others(*args, **kwargs)
        self.hook_manager.add_hook(
            lambda: ext.trainer.
            change_with_training_progress(self.model, self.step, self.num_steps, self.epoch, self.num_epochs),
            'before_train_step'
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
        if self.cfg.mesh and hasattr(self.model, 'extract_geometry'):
            save_path = self.output.joinpath(self.cfg.mesh)
            assert save_path.suffix in utils.mesh_extensions
            # Ts = getattr(self.train_db, 'Ts', None) if self.cfg.mesh_scale else None
            vertices, triangles = self.model.extract_geometry(self.cfg.resolution)
            self.logger.info(f"Extrace Geometry have {len(vertices)} vertices, {len(triangles)} triangles")
            utils.save_mesh(save_path, vertices, triangles)

            self.logger.info(f'==> Save mesh (resoution={self.cfg.resolution}) to {save_path}')
            return
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
        self.progress.start('train')
        self.model.train()
        self.hook_manager.before_train_step()
        inputs, targets, infos = ext.utils.tensor_to(data, device=self.device)
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
            loss_dict = self.criterion(inputs, outputs, targets)
            if self.cfg.debug:
                self.logger.debug(f'loss_dict: {loss_dict}')
            losses = sum(loss_dict.values())  # type: Tensor # noqa
            self.execute_backward(losses)
        self.loss_dict_meter.update(loss_dict)
        self.losses_meter.update(losses)

        with torch.no_grad():
            if 'mse' in loss_dict:
                mse = loss_dict['mse']
            else:
                mse = F.mse_loss(outputs['images'][..., :3], targets['images'][..., :3])
            psnr = -10 * torch.log10(mse)
            if self.cfg.weighted_sample:
                self.train_db.update_errors(outputs['images'], targets['images'])
        self.psnr_meter.update(psnr)
        self.hook_manager.after_train_step()

        if ext.utils.check_interval(self.step, self.cfg.print_f, self.num_steps):
            self.logger.info(
                f"[{self.step}]/[{self.num_steps}]: "
                f"loss={utils.float2str(self.losses_meter.avg)}, {self.loss_dict_meter.average}, "
                f"psnr={utils.float2str(self.psnr_meter.avg)}, "
                f"lr={utils.float2str(self.lr_scheduler.get_lr(self.cfg.lr))}, "
                f"{self.train_timer.progress}",
            )
            self.psnr_meter.reset()
            self.loss_dict_meter.reset()
            self.losses_meter.reset()

        self.progress.step('train')
        self.visualize()

    def split_data(self, shape, *data: dict, batch_size=512):
        if batch_size <= 0:
            yield utils.tensor_to(*data, device=self.device, non_blocking=True)
            return
        n_dim = len(shape)
        total = np.prod(shape)
        # logging.debug(f"split data: shape: {shape}, n_dim: {n_dim}, total={total}, bs={batch_size}")

        for i in range(0, total, batch_size):
            s, e = i, min(total, i + batch_size)

            def split(v):
                if isinstance(v, Tensor) and v.ndim >= n_dim and v.shape[:n_dim] == shape:
                    return v.flatten(0, n_dim - 1)[s:e]
                else:
                    return v

            outoputs = utils.make_recursive_func(split)(data)
            if len(data) == 1:
                outoputs = outoputs[0]
            yield utils.tensor_to(outoputs, device=self.device, non_blocking=True)

    def concat_output(self, shape, *outputs, batch_size=512):
        if batch_size <= 0:
            results = outputs[0]
            return results[0] if len(results) == 1 else results
        results = []
        for output in outputs:
            output = torch.concat(output, dim=0)
            output = output.reshape(*shape, *output.shape[1:])
            results.append(output)
        return results[0] if len(results) == 1 else results

    @torch.no_grad()
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
            self.hook_manager.before_eval_step()
            self.logger.debug(f'inputs: {utils.show_shape(data[0])}')
            self.logger.debug(f'targets: {utils.show_shape(data[1])}')
            self.logger.debug(f'infos: {utils.show_shape(data[2])}')
            pred_images = []
            num_each_batch = []
            loss_dict_list = []
            shape = data[0]['rays_o'].shape[:-1] if 'rays_o' in data[0] else data[2]['size']
            for inputs, targets, infos in self.split_data(shape, *data, batch_size=batch_size):
                # self.logger.debug(f"split: {utils.show_shape(inputs, targets, infos)}")
                num_each_batch.append(inputs['rays_o'].shape[0] if 'rays_o' in inputs else 1)
                with ext.autocast(self.cfg.fp16):
                    if hasattr(self.model, 'render'):
                        outputs = self.model.render(**inputs, **eval_kwargs, info=infos)
                    else:
                        outputs = self.model(**inputs, **eval_kwargs, info=infos)
                # self.logger.debug(f'outputs: {utils.show_shape(outputs)}')
                loss_dict_list.append(self.criterion(inputs, outputs, targets))
                pred_images.append(outputs['images'])
                del outputs
            # weight loss_dict
            total = sum(num_each_batch)
            loss_dict = {}
            for k in loss_dict_list[0].keys():
                loss_dict[k] = sum([n * v[k] for n, v in zip(num_each_batch, loss_dict_list)]) / total
            self.logger.debug(f'losses: {loss_dict}')
            pred_images = self.concat_output(shape, pred_images, batch_size=batch_size)
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

            self.progress.step('eval', self.metric_manager.str())
            self.hook_manager.after_eval_step()
            if self.mode == 'train' and 0 < self.cfg.eval_num_steps <= step:
                break
            if self.cfg.debug:
                break
        self.hook_manager.after_eval_epoch()
        if self.cfg.optim_camera_pose:
            R_error, t_error = ops_3d.camera_poses_error(*self.model.get_poses())
        else:
            R_error, t_error = None, None
        self.logger.info("Eval [%d/%d]: %s%s", self.step, self.num_steps, self.metric_manager.str(),
            f", R_err={utils.float2str(R_error.item())}°, t_err={utils.float2str(t_error.item())},"
            if self.cfg.optim_camera_pose else ''
        )
        if self.mode == 'train':
            if self.metric_manager.is_best:
                self.save_model('best.pth')
            self.progress.stop('eval')
        return

    @torch.no_grad()
    def test(self):
        db = self.test_db
        W, H = db.image_size
        K = db.Tv2s

        render_poses = db.render_poses if hasattr(db, 'render_poses') else get_poses(None, -30.0, 4.0, 40)
        self.progress.start('test', len(render_poses))

        rgbs = []
        disps = []

        for i, c2w in enumerate(render_poses):
            rays_o, rays_d = ops_3d.get_rays(K, c2w[:3, :4], size=(W, H), offset=0)
            rays_o = rays_o.flatten(0, -2).cuda()
            rays_d = rays_d.flatten(0, -2).cuda()
            image = []
            for rays_o_part, rays_d_part in self.split_data([H, W], rays_o, rays_d, batch_size=512):
                outputs = self.model.render(
                    rays_o=rays_o_part,
                    rays_d=rays_d_part,
                    viewdirs=None,
                    # near=db.near,
                    # far=db.far,
                    # ndc=db.ndc,
                    t=torch.full_like(rays_o_part[..., 0], i / len(render_poses)),
                    # hwf=db.hwf,
                    **self.cfg.test_kwargs,
                    # verbose=False,
                )
                image.append(outputs['images'])
                del outputs
            # print(f'out[{i}]: {ext.utils.show_shape(outputs)}')
            image = self.concat_output([H, W], image)
            image = image.clamp(0, 1) * 255.
            image = image.cpu().numpy().astype(np.uint8).reshape(H, W, 3)
            rgbs.append(image)
            self.progress.step(task='test')
            if self.cfg.debug:
                break
        rgbs = np.stack(rgbs)
        if self.cfg.save_video:
            ext.utils.save_video(self.output.joinpath(f"test_{self.cfg.exp_name}.gif"), rgbs)
            return
        self.progress.stop('test')
        return

    @torch.no_grad()
    def visualize(self):
        if not utils.check_interval(self.step, self.cfg.vis_interval):
            return
        self.model.eval()
        torch.cuda.empty_cache()
        vis_kwargs = self.cfg.vis_kwargs.copy()  # type: dict
        batch_size = self.cfg.batch_size[1]
        self.progress.pause('train')
        index = np.random.randint(0, len(self.train_db))
        logging.info(f"visualize image {index} as step {self.step}")
        inputs, targets, info = self.train_db[index]
        self.logger.debug(f'inputs: {utils.show_shape(inputs)}')
        self.logger.debug(f'targets: {utils.show_shape(targets)}')
        self.logger.debug(f'info: {utils.show_shape(info)}')
        images = []
        shape = inputs['rays_o'].shape[:-1] if 'rays_o' in inputs else info['size']
        info = utils.tensor_to(info, device=self.device)
        for inputs_i in self.split_data(shape, inputs, batch_size=batch_size):
            if hasattr(self.model, 'render'):
                outputs_i = self.model.render(**inputs_i, **vis_kwargs, info=info)
            else:
                outputs_i = self.model(**inputs_i, **vis_kwargs, info=info)
            images.append(outputs_i['images'] if 'images' in outputs_i else None)
            del outputs_i
        cat_dim = 0 if self.train_db.aspect > 1. else 1
        if images[0] is not None:
            img_pred = self.concat_output(shape, images, batch_size=batch_size)[..., :3].cpu()
            img_gt = targets['images'][..., :3].cpu()
            if img_pred.ndim == 4:
                assert img_pred.shape[0] == 1
                img_pred = img_pred[0]
            if img_pred.ndim == 3:
                image = torch.cat([img_pred, img_gt], dim=cat_dim)
            else:
                W, H = self.train_db.image_size
                image = torch.ones((H * 2, W, 3)) if cat_dim == 0 else torch.ones((H, W * 2, 3))
                points = ops_3d.xfm(ops_3d.xfm(inputs['rays_o'] + inputs['rays_d'], info['Tw2v'].cpu()),
                    info['Tv2s'].cpu())
                points = (points[..., :2] / points[..., 2:3]).int()
                image[points[:, 1], points[:, 0], :] = img_pred
                H, W = (H, 0) if cat_dim == 0 else (0, W)
                image[points[:, 1] + H, points[:, 0] + W, :] = img_gt
            utils.save_image(self.output.joinpath('vis', f"img_{self.step}_{index}.png"), image)
        if hasattr(self.model, 'extract_geometry'):
            vertices, triangles = self.model.extract_geometry(64)
            self.logger.info(f"Extrace Geometry have {len(vertices)} vertices, {len(triangles)} triangles")
            utils.save_mesh(self.output.joinpath('vis', f'mesh{64}_{self.step}.obj'), vertices, triangles)
        self.model.train()


if __name__ == '__main__':
    NeRFTask().run()
