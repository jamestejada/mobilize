import logging
import logging.handlers
from datetime import datetime

from .settings import LOG_DIR


def setup_logging() -> None:
    log_file = LOG_DIR / f"bot{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Suppress verbose debug output from third-party libraries
    for noisy in (
            "pydantic_ai",
            "openai",
            "hpack",
            "rquest",
            "cookie_store",
            "httpcore.http11",
            "httpcore.http2",
            "httpcore.connection",
            "primp.utils",

        ):
        logging.getLogger(noisy).setLevel(logging.WARNING)