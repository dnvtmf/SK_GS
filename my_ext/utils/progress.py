from typing import Any

from rich import get_console
from rich.progress import (
    Progress as Projress_rich,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    ProgressColumn,
    Task, Text,
)

__all__ = ['Progress']


class MessageColumn(ProgressColumn):
    def render(self, task: "Task") -> Text:
        return Text(str(task.fields['message']) if 'message' in task.fields else '')


class Progress:
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        self.progress = Projress_rich(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn("-->"),
            TimeElapsedColumn(),
            MessageColumn(),
            console=get_console(),
            disable=not enable,
            # speed_estimate_period=3600,
        )
        self.tasks = {}
        for sub_name, counts in kwargs.items():
            if counts is None:
                continue
            if not isinstance(counts, int):
                counts = len(counts)
            if counts > 0:
                self.tasks[sub_name] = self.progress.add_task(sub_name, total=counts, start=False, visible=False)

    def add_task(self, name: str, total: int, completed=0, **kwargs):
        self.tasks[name] = self.progress.add_task(
            name,
            total=total,
            start=False,
            visible=False,
            completed=completed,
            **kwargs
        )

    def set_completed(self, task, completed: int = 0):
        self.progress.update(self.tasks[task], completed=completed)

    def start(self, task, total=None, **kwargs):
        if task not in self.tasks and total is not None:
            self.tasks[task] = self.progress.add_task(task, total=total, **kwargs)
            return
        self.progress.start_task(self.tasks[task])
        self.progress.update(self.tasks[task], visible=True, total=total, **kwargs)

    def stop(self, task):
        if task is None:
            self.progress.stop()
            return
        self.progress.stop_task(self.tasks[task])
        self.progress.update(self.tasks[task], visible=False)

    def pause(self, task):
        self.progress.stop_task(self.tasks[task])

    def reset(self, task, start=True, **kwargs):
        self.progress.reset(self.tasks[task], start=start, visible=False, **kwargs)
        self.progress.tasks[self.tasks[task]].stop_time = None

    def step(self, task, msg: Any = '', step=1):
        msg = str(msg)
        self.progress.advance(self.tasks[task], step)
        if msg:
            self.progress.update(self.tasks[task], message=msg)

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()


if __name__ == '__main__':
    import time

    _p = Progress(all=10, train=30, eval=20)
    with _p:
        _p.start('all')
        for _i in range(10):
            # train
            _p.reset('train', start=False)
            _p.start('train')
            for _j in range(30):
                time.sleep(0.2)
                _p.step('train', f"step={_j}", 1)

            _p.stop('train')
            # eval
            _p.reset('eval', start=False)
            _p.start('eval')
            for _j in range(20):
                time.sleep(0.1)
                _p.step('eval')
            _p.stop('eval')
            _p.step('all')
