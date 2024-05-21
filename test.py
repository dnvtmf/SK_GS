import json

import torch
from tqdm import tqdm

from train import GaussianTrainTask
from my_ext import utils
from datasets.base import DynamceSceneDataset


class SuperpointGaussianTestTask(GaussianTrainTask):
    def extra_config(self, parser):
        super().extra_config(parser)
        utils.add_bool_option(parser, '--fps', default=False)
        utils.add_cfg_option(parser, '--lpips', default=True)
        utils.add_cfg_option(parser, '--lpips-alex', default=False)
        utils.add_cfg_option(parser, '--lpips-vgg', default=False)
        utils.add_cfg_option(parser, '--ssim', default=True)
        utils.add_cfg_option(parser, '--ms-ssim', default=True)
        parser.set_defaults(test=True)

    def step_2_environment(self, *args, **kwargs):
        metric = 'image/PSNR'
        if self.cfg.ssim:
            metric = metric + '/SSIM'
        if self.cfg.ms_ssim:
            metric = metric + '/MS_SSIM'
        if self.cfg.lpips or self.cfg.lpips_alex:
            metric = metric + '/LPIPS'
        if self.cfg.lpips_vgg:
            metric = metric + '/LPIPS_VGG'
        self.cfg.metrics = metric
        super().step_2_environment(*args, **kwargs)

    @torch.no_grad()
    def run(self):
        self.model.eval()
        save_images = []
        save_images_c = []
        if self.cfg.test:
            db = self.test_db  # noqa
            self.logger.info('using test split')
        elif self.cfg.eval:
            db = self.eval_db  # noqa
            self.logger.info('using eval split')
        else:
            db = self.train_db  # noqa
            self.logger.info('using train split')
        db: DynamceSceneDataset
        times = db.times.cuda()
        Tw2v = db.Tw2v.cuda()
        Tw2c = db.Tv2c.cuda() @ Tw2v
        campos = db.Tv2w[:, :3, 3].cuda()
        background = torch.zeros((1, 1, 3)).cuda() if db.background_type != 'white' else torch.ones((1, 1, 3)).cuda()
        start_time = torch.cuda.Event(enable_timing=True)
        start_time.synchronize()
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        for i in tqdm(range(len(db))):
            if hasattr(db, 'camera_ids') and db.num_cameras > 1:
                cid = db.camera_ids[i].item()
            else:
                cid = i
            outputs = self.model.render(info={
                'Tw2v': Tw2v[cid],
                'Tw2c': Tw2c[cid],
                'campos': campos[cid],
                'size': db.image_size,
                'FoV': db.FoV[cid] if db.FoV.ndim == 2 else db.FoV,
            }, t=times[i], background=background)
            save_images.append(outputs['images'].clamp(0, 1))
            if 'images_c' in outputs:
                save_images_c.append(outputs['images_c'])
        end_time.record()
        end_time.synchronize()
        self.logger.info(f"[red]FPS: {len(times) * 1000 / start_time.elapsed_time(end_time)}")
        self.logger.info('Begin evalution')
        save_dir = self.output.joinpath('test')
        utils.dir_create_and_clear(save_dir, "*.png")
        for i in tqdm(range(len(times))):
            gt_img = db.get_image(i)[..., :3]
            pd_img = save_images[i]
            self.metric_manager.update('image', pd_img, gt_img)
            utils.save_image(save_dir.joinpath(f"{i:04d}.png"), pd_img)
        results = self.metric_manager.summarize()['image']
        self.logger.info(results)

        if len(save_images_c) > 0:
            self.metric_manager.reset()
            for i in tqdm(range(len(times))):
                gt_img = db.images[i, :, :, :3]
                pd_img = save_images_c[i]
                self.metric_manager.update('image', pd_img, gt_img)
                utils.save_image(save_dir.joinpath(f"{i:04d}.png"), pd_img)
            self.logger.info(f"images_c: {self.metric_manager.summarize()}")

        if self.cfg.fps:
            start_time.synchronize()
            start_time.record()
            for i in tqdm(range(1000)):
                outputs = self.model.render(info={
                    'Tw2v': Tw2v[0],
                    'Tw2c': Tw2c[0],
                    'campos': campos[0],
                    'size': db.image_size,
                    'FoV': db.FoV[0] if db.FoV.ndim == 2 else db.FoV,
                }, t=times.new_tensor([i / 1000]), background=background)
                save_images.append(outputs['images'])
            end_time.record()
            end_time.synchronize()
            FPS = 1000 * 1000 / start_time.elapsed_time(end_time)
            self.logger.info(f"[red]FPS: {FPS}")
            results['FPS'] = FPS
        with self.output.joinpath('results.json').open('w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    SuperpointGaussianTestTask().run()
