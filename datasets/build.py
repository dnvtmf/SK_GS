import logging
from pathlib import Path

import my_ext as ext
from my_ext import utils
from my_ext.config import get_parser
from my_ext.utils import eval_str, to_list, extend_list, add_path_option
from my_ext.utils.registry import Registry
from datasets.base import NERF_DATASETS, NERF_DATASET_STYLE

_is_new_datasets_added = False


def add_or_update_datasets(new_datasets: dict, DATASETS: Registry):
    global _is_new_datasets_added
    if _is_new_datasets_added:
        return
    if new_datasets is None:
        return
    if isinstance(new_datasets, dict):
        new_datasets = [new_datasets]
    for new_dataset in new_datasets:  # type: dict
        assert isinstance(new_dataset, dict)
        name = new_dataset['name']
        if name in DATASETS:
            DATASETS[name].update(new_dataset)
            logging.info(f'Update the datasets: {name}')
        else:
            DATASETS[name] = new_dataset
            logging.info(f'Add new datasets: {name}')
    _is_new_datasets_added = True
    return


def options(parser=None):
    group = get_parser(parser).add_argument_group('Datset Options')
    add_path_option(group, '--dataset-root', default='~/data', help='The root directory of dataset.')
    group.add_argument('--dataset', default=None, type=eval_str,
                       help='The datasets use in train/eval/test. '
                            'Format: name, name/split, [name/split, ...], [{name: name/split, **cfg}]. '
                            f'Default datastes: {list(NERF_DATASETS.keys())}'
                       )
    group.add_argument('--dataset-train', default=None, type=eval_str, help='The dataset use for train')
    group.add_argument('--dataset-eval', default=None, type=eval_str, help='The dataset use for eval')
    group.add_argument('--dataset-test', default=None, type=eval_str, help='The dataset use for test')
    ext.utils.add_cfg_option(group, "--dataset-cfg", default={}, help="The common configure for all datasets")
    group.add_argument("--new-datasets", default=None, metavar='E', type=eval_str,
                       help=f"Add new datasets. Styles: {list(NERF_DATASET_STYLE.keys())}")
    group.add_argument('--scene', default='', help='The name of scene')
    utils.add_bool_option(group, '--weighted-sample', default=False)
    return group


def make(cfg, mode='train', **kwargs):
    DATASETS = NERF_DATASETS
    kwargs.setdefault('scene', cfg.scene)
    # kwargs.setdefault('weighted_sample', cfg.weighted_sample and mode == 'train')
    add_or_update_datasets(cfg.new_datasets, DATASETS)
    assert mode in ['train', 'eval', 'test']
    split_idx = ['train', 'eval', 'test'].index(mode)
    if mode == 'train':
        names_and_cfgs = cfg.dataset_train
    elif mode == 'eval':
        names_and_cfgs = cfg.dataset_eval
    else:
        names_and_cfgs = cfg.dataset_test
    if names_and_cfgs is None or not names_and_cfgs:
        names_and_cfgs = cfg.dataset

    datasets = []
    root = Path(cfg.dataset_root).expanduser()

    def obtain_datasets(dataset_name: str, configurations: dict):
        dataset_name, dataset_split = extend_list(dataset_name.split('/'), 2)
        assert dataset_split in DATASETS[dataset_name], ValueError(f'{dataset_split} is not a part of {dataset_name}')
        splits = DATASETS[dataset_name][dataset_split]

        if isinstance(splits, dict):
            splits = [dataset_split]
        elif isinstance(splits, list):
            splits = to_list(splits[min(split_idx, len(splits) - 1)])
        else:
            raise ValueError('Unknown how to deal such situation!')

        for name in splits:
            # 数据集参数优先度: kwargs > dataset_name中的配置 > cfg.dataset_cfg > 数据集中的配置  > 数据集类的默认参数
            parameters = DATASETS[dataset_name].get('common', {}).copy()
            if name in DATASETS[dataset_name]:
                parameters.update(DATASETS[dataset_name][name])
            style = parameters.pop("style")
            parameters.update(cfg.dataset_cfg)
            parameters.update(configurations)
            parameters.update(kwargs)
            parameters['mode'] = mode

            root_ = parameters.get("root", None)
            if root_ is not None:
                root_ = Path(root_).expanduser()
                if root_.is_absolute():
                    parameters["root"] = root_
                else:
                    parameters["root"] = root / root_
            else:
                parameters["root"] = root

            dataset = NERF_DATASET_STYLE[style](**parameters)
            datasets.append(dataset)
        return

    if isinstance(names_and_cfgs, str):
        obtain_datasets(names_and_cfgs, {})  # 生成方式1,2,3
    elif isinstance(names_and_cfgs, list):
        for name_and_cfg in names_and_cfgs:
            if isinstance(name_and_cfg, str):
                obtain_datasets(name_and_cfg, {})  # 生成方式 4
            elif isinstance(name_and_cfg, dict):
                obtain_datasets(name_and_cfg['name'], name_and_cfg)  # 生成方式 6
            else:
                raise ValueError('name_and_cfg must be str or dict')
    elif isinstance(names_and_cfgs, dict):
        for name_, cfg_ in names_and_cfgs.items():
            assert isinstance(name_, str) and isinstance(cfg_, dict)
            obtain_datasets(name_, cfg_)  # 生成方式 5
    else:
        raise ValueError('Error to obtain datasets')

    assert len(datasets) == 1
    return datasets[0]
