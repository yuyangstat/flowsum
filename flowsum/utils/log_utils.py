"""Set loggers for the files inside the nfsummary module.

References: 
(1) https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/utils/logging.py
"""
import logging
import os
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the nfsummary's root logger.
    Args:
        verbosity (`int`): logging level
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def _get_library_name() -> str:

    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def _get_default_logging_level():
    """
    If NFSUMMARIZER_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("NFSUMMARIZER_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option NFSUMMARIZER_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter(
            "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        )
        handler.setFormatter(formatter)
