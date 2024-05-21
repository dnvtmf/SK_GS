import logging

__all__ = ['let_trimesh_no_warn']


def let_trimesh_no_warn():

    class TrimeshWarningFilter(logging.Filter):

        def filter(self, record) -> bool:
            if record.levelno == logging.WARNING:
                return False
            return True

    logging.getLogger('trimesh').addFilter(TrimeshWarningFilter())
