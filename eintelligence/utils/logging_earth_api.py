from __future__ import annotations
import json, os, sys, time
import logging
import logging.handlers
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any

DEFAULT_LEVEL = os.getenv("EARTH_LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.getenv("EARTH_LOG_DIR", Path(__file__).resolve().parents[2] / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

class JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter for log records.
    """
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in getattr(record, "__dict__", {}).items():
            if key in ("args", "msg", "levelname", "levelno", "name", "created", "msecs", 
                       "relativeCreated", "pathname", "filename", "module", "exc_info", "exc_text",
                       "stack_info", "lineno", "funcName", "thread", "threadName", "process", "processName", "asctime"):
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)  # test serializability
                payload[key] = value
            except Exception:
                payload[key] = repr(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)
    
def _make_console_handler(level: int, pretty: bool) -> logging.Handler:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    if pretty:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt))
    else:
        console_handler.setFormatter(JsonFormatter())
    return console_handler

def _make_file_handler(path: Path, level: int, json_mode: bool) -> logging.Handler:
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=20_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter() if json_mode else logging.Formatter(
        "%(asctime)s | %(levelname) | %(name)s | %(message)s"))
    return file_handler

def setup_logger(
    *,
    level: str = DEFAULT_LEVEL,
    json_console: bool = False,
    file_json: bool = True,
    file_name: str = "earth_api.log"
) -> None:
    """
    Initialize root logging exactly once. Safe to call multiple times.
    """
    if getattr(setup_logger, "_configured", False):
        return
    setup_logger._configured = True

    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(lvl)

    # Console
    pretty_dev = (not json_console) and (os.getenv("EARTH_ENV", "dev") != "prod")
    root.addHandler(_make_console_handler(lvl, pretty_dev))

    # File
    log_path = LOG_DIR / file_name
    root.addHandler(_make_file_handler(log_path, lvl, file_json))

    # quiet some noisy loggers
    logging.getLogger("rasterio").setLevel(max(lvl, logging.WARNING))
    logging.getLogger("urllib3").setLevel(max(lvl, logging.WARNING))

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    """
    return logging.getLogger(name)

@contextmanager
def log_timed(logger: logging.Logger, message: str, **ctx):
    """
    Context manager to log the time taken by a block.
    """
    start = time.perf_counter()
    logger.info(message + " - started", **ctx)
    try:
        yield
    except Exception:
        logger.exception(message + " - failed", **ctx)
        raise
    finally:
        duration = (time.perf_counter() - start) * 1000.0
        logger.info(message + f" - completed in {duration:.2f} ms", **ctx)