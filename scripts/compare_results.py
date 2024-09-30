import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.table import Table

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='./results', type=str)
parser.add_argument('-m', '--mertic', default='', type=str)
args = parser.parse_args()

console = Console()
root = Path(args.input).expanduser()
result_paths = list(root.rglob('results.json'))
## extract data
results = {}
methods = defaultdict(list)
scene_to_method = {}
metrics = set()
num_root_parts = len(root.parts)
for result_path in result_paths:
    with open(result_path, 'r') as f:
        res = json.load(f)  # type: Dict[str, float]
        parts = result_path.parts[num_root_parts:]
        if len(parts) == 4:
            method, scene, vary = parts[:3]
            method = f'{method}/{vary}'
        else:
            method, scene = parts[:2]
        if scene not in scene_to_method:
            scene_to_method[scene] = method
        methods[method].append(scene)
        results[(method, scene)] = res
        metrics = metrics.union(res.keys())
print(f"Find metrics:", metrics)

## display result
if args.mertic:
    if args.mertic not in metrics:
        exit()
    metrics = [args.mertic]
for metric in metrics:  # type: str
    visited = {method: False for method in methods.keys()}
    for method, vis in visited.items():
        if vis:
            continue
        all_scenes = set(methods[method])
        all_methods = []
        num_methods = len(all_methods)
        updated = False
        while not updated:
            updated = False
            for method_, scenes_ in methods.items():
                # if not visited[method_] and len(all_scenes.intersection(scenes_)) > 0:
                if not visited[method_] and all_scenes == set(scenes_):
                    all_scenes.update(scenes_)
                    all_methods.append(method_)
                    updated = True
        all_scenes = sorted(list(all_scenes))
        all_methods = sorted(list(all_methods))

        table = Table()
        table.add_column(metric, justify="left")
        for scene in all_scenes:  # type: str
            table.add_column(scene, justify="center")
        table.add_column('avg', justify="center")
        num_methods = 0
        for method_ in all_methods:
            visited[method_] = True
            num_scenes = 0
            row = [method_]
            sum_metric = 0
            for scene_ in all_scenes:  # type: str
                res = results.get((method_, scene_), None)
                if res is not None and metric in res:
                    value = res[metric]
                    sum_metric += value
                    num_scenes += 1
                    row.append(f"{value:.4f}")
                else:
                    row.append("-")
            if num_scenes > 0:
                row.append(f"{sum_metric / num_scenes:.4f}")
                table.add_row(*row)
                num_methods += 1
        if num_methods > 0:
            console.print(table)
