import logging
from collections import defaultdict
from typing import Dict, Tuple

from my_ext.config import get_parser
from my_ext.utils import add_cfg_option, add_choose_option, float2str
from .base import METRICS, Metric


def options(parser=None):
    group = get_parser(parser).add_argument_group('Metric Options')
    group.add_argument(
        '--metric', default='loss', help='The metric for evalutation. Format: NAME, NAME/ITEM. default: loss'
    )
    add_cfg_option(
        group,
        '--metrics',
        default=['loss'],
        help='The metric for evalutation. one or list of these format:'
             'TYPE/ITEM1/ITEM2/..., {type: TYPE, name: NAME, items: [ITEM1, ...], **cfg}'
    )
    add_choose_option(
        group,
        '--metric-goal',
        default='minimize',
        choices=['maximize', 'minimize', 'max', 'min', '+', '-'],
        help='To get best model, the value of metric should be minimzed or minimized?'
    )
    return group


class MetricManager(Metric):

    def __init__(self, metrics: Dict[str, Metric] = None, main: Tuple[str, str] = None, is_maximize=True):
        self.is_maximize = is_maximize
        self.main_metric = (None, None) if main is None else main
        self.metrics = {} if metrics is None else metrics
        self.best_score = None
        self.is_best = False
        assert len(self.main_metric) == 2

    def reset(self):
        self.is_best = False
        self.best_score = None
        for name, metric in self.metrics.items():
            metric.reset()

    def have_metric(self, *names: str):
        return any(name in self.metrics for name in names)

    def state_dict(self):
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict):
        self.best_score = state_dict['best_score']

    def update(self, name, *args, **kwargs):
        if name in self.metrics:
            self.metrics[name].update(*args, **kwargs)
        else:
            logging.debug(f'name "{name}" not in metrics: {list(self.metrics.keys())}')

    def __getitem__(self, name: str):
        if name in self.metrics:
            return self.metrics[name]
        for name, metric in self.metrics.items():
            if name in metric.names:
                return metric[name]

    def summarize(self):
        results = defaultdict(dict)
        for name, metric in self.metrics.items():
            metric.summarize()
            for item in metric.names:
                score = metric.get_result(item)
                results[name][item] = score
        # update main metric
        self.is_best = False
        main_name, main_item = self.main_metric
        if main_name is not None:
            main_item = main_item if main_item is not None else self.metrics[main_name].names[0]
            score = results[main_name][main_item]
            if self.best_score is None:  # 首次更新
                self.best_score = score
                self.is_best = True
            elif self.is_maximize and score > self.best_score:  # 最大化
                self.best_score = score
                self.is_best = True
            elif not self.is_maximize and score < self.best_score:  # 最小化
                self.best_score = score
                self.is_best = True
        return dict(results)

    def str(self, name=None, fmt=float2str, **kwargs):
        return '; '.join(metric.str(name=name, fmt=fmt, **kwargs) for metric in self.metrics.values())

    def __repr__(self):
        s = f"{'Maximize' if self.is_maximize else 'Minimize'} metric: {self.main_metric[0]}/{self.main_metric[1]}\n"
        s += f"Metrics: \n"
        for name, metric in self.metrics.items():
            s += f"  {name}: {metric}\n"
        return s[:-1]


def make(cfg, **kwargs):
    def parse_str(s: str):
        items = [item.strip() for item in s.split('/') if item]
        assert len(items) > 0, f"metric and sub item splited by /, now '{s}' is error"
        return {'type': items[0], 'items': None if len(items) == 1 else items[1:]}

    metrics = {}
    for metric_cfg in cfg.metrics if isinstance(cfg.metrics, (tuple, list)) else [cfg.metrics]:
        if isinstance(metric_cfg, str):  # 格式1: 为单个字符串: TYPE/ITEM1/ITEM2/...
            metric_cfg = parse_str(metric_cfg)
        m_type = metric_cfg.pop('type')
        name = metric_cfg.pop('name', m_type)
        assert name not in metrics
        metric = METRICS[m_type](**metric_cfg)
        metrics[name] = metric

    if cfg.metric is None:
        main_name, main_item = None, None
    else:
        main_metric_cfg = parse_str(cfg.metric)
        main_name = main_metric_cfg['type']
        main_item = main_metric_cfg['items']
        if main_item is not None:
            main_item = main_item[0]
    if main_name is None:
        main_name = list(metrics.keys())[0] if len(metrics) > 0 else None
    elif main_name not in metrics:
        metrics[main_name] = METRICS[main_name](items=main_item)

    return MetricManager(metrics, (main_name, main_item), is_maximize=cfg.metric_goal in ['maximize', 'max', '+'])
