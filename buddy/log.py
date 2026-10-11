"""
Logging. Every buddy module logs to the "buddy" logger (`log = logging.getLogger("buddy")`); nothing configures
Python's root logger any more, so buddy no longer takes over the logging of notebooks or other packages.

    from buddy.log import setup_logging
    setup_logging()                  # messages on the console, as the notebooks expect (called by Experiment)
    setup_logging(level="WARNING")   # quieter
    setup_logging(file="run.log")    # also to a file
"""
import logging
import sys
from contextlib import contextmanager

FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S%p"

log = logging.getLogger("buddy")


def setup_logging(level="INFO", file=None, force=False):
    """Console (+ optional file) output for buddy's messages. Does nothing when already set up, unless force."""
    if log.handlers and not force:
        return log
    for h in list(log.handlers):
        log.removeHandler(h)
    formatter = logging.Formatter(FORMAT, DATE_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)
    if file:
        fh = logging.FileHandler(file)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    log.setLevel(level)
    log.propagate = False
    return log


@contextmanager
def keep_logging():
    """
    cellpose_omni re-configures Python's root logger when it is imported (own format, own handler), which changes
    the look of every later buddy message. Import / load Omnipose inside this block to keep buddy's logging as it was.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        yield
    finally:
        for h in list(root.handlers):
            if h not in handlers:
                root.removeHandler(h)
        for h in handlers:
            if h not in root.handlers:
                root.addHandler(h)
        root.setLevel(level)
        logging.getLogger("cellpose_omni").setLevel(logging.WARNING)      # its model-setup chatter is not buddy output
