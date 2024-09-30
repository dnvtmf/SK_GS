import json
import os
from pathlib import Path
import rich
from rich.console import Console
from rich.table import Table
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='./results', type=str)
args = parser.parse_args()

console = Console()
root = Path(args.input).expanduser()
datsets = list(root.iterdir()) + [root]
for db in sorted(datsets):
    scenes = [scene for scene in sorted(os.listdir(db)) if db.joinpath(scene, 'results.json').exists()]
    if len(scenes) == 0:
        continue
    table = Table(title=f"Results for {db.name}")
    table.add_column()
    results = []
    num_valid = 0
    for scene in scenes:
        with open(db.joinpath(scene, 'results.json'), 'r') as f:
            res = json.load(f)
            results.append(res)
            num_valid += 1
        table.add_column(scene)
    if num_valid == 0:
        continue
    table.add_column('average')
    for metric in ['PSNR', 'SSIM', 'MS-SSIM', "LPIPS (VGG)", 'LPIPS', 'FPS']:
        row = [metric]
        total = 0
        num = 0
        for res_s in results:
            # print(res_s)
            if metric in res_s:
                row.append('{:.4f}'.format(res_s[metric]))
                total += res_s[metric]
                num += 1
            else:
                row.append('-')
        if total == 0:
            row.append('N/A')
        else:
            row.append('{:.4f}'.format(total / num))
        table.add_row(*row)
    console.print(table)
