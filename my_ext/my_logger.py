import argparse
import logging
import shutil
from pathlib import Path

from rich.logging import RichHandler
from rich.traceback import install

from .config import get_parser
from .utils import add_bool_option

__all__ = ['options', 'make', 'logger', 'get_logger']
install()


def options(parser: argparse.ArgumentParser = None):
    group = get_parser(parser).add_argument_group('Logging Options')
    add_bool_option(group, '--no-log', default=False, help='Do not write log to file')
    add_bool_option(group, '--no-print', default=False, help='Do not print log to stdout')
    group.add_argument("--log-filename", default='', help='The filename of log')
    group.add_argument("--log-suffix", metavar="S", default="", help="the suffix of log path.")
    group.add_argument("--print-f", metavar="N", default=100, type=int, help="print frequency. (default: 100)")
    add_bool_option(group, "--debug", default=False, help="Debug this program?")
    return group


logger = None
_log_file: Path = None


class LogLevelFilter(logging.Filter):
    def __init__(self, low=logging.NOTSET, high=logging.CRITICAL):
        super().__init__()
        self.low = low
        self.high = high

    def filter(self, record: logging.LogRecord) -> bool:
        return self.low <= record.levelno <= self.high


class MyFormater(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if record.levelno >= logging.CRITICAL:
            s = "[bold red]" + s
        elif record.levelno >= logging.ERROR:
            s = "[red]" + s
        elif record.levelno >= logging.WARNING:
            s = "[green]" + s
        return s


class DebugFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).parent.parent.as_posix()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == 'root' and 'python' not in record.pathname


def get_logger():
    return logger


def basic_config(debug=False, no_print=False, enable=True):
    global logger
    logger = logging.getLogger()
    if not enable:
        logger.setLevel(logging.WARNING)
        return logger
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # remove handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    if no_print:
        return
    if debug:
        debug_handler = RichHandler()
        debug_handler.addFilter(LogLevelFilter(high=logging.DEBUG))
        debug_handler.addFilter(DebugFilter())
        logger.addHandler(debug_handler)
    info_hander = RichHandler(level=logging.INFO, show_level=False, show_path=False, show_time=False, markup=True)
    info_hander.addFilter(LogLevelFilter(logging.INFO, logging.INFO))
    logger.addHandler(info_hander)

    warn_error_hander = RichHandler(level=logging.WARN, markup=True)
    warn_error_hander.addFilter(LogLevelFilter(logging.WARN))
    warn_error_hander.setFormatter(MyFormater())
    logger.addHandler(warn_error_hander)
    return logger


def make(cfg, output_dir: Path = None, log_filename='log.txt', enable=True):
    global _log_file, logger
    logger = basic_config(cfg.debug, cfg.no_print, enable)
    if enable and not cfg.no_log and not cfg.debug:
        if output_dir is None:
            output_dir = Path('.')
        _log_file = output_dir.joinpath(log_filename)
        file_handler = logging.FileHandler(_log_file, mode='a' if getattr(cfg, 'resume', '') else 'w')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


def copy_to(dst=None, prefex='', suffix='', ext=''):
    dst = _log_file if dst is None else dst
    if dst is None:
        return
    dst = dst.with_name(f"{prefex}{dst.stem}{suffix}{ext if ext else dst.suffix}")
    shutil.copy(_log_file, dst)
