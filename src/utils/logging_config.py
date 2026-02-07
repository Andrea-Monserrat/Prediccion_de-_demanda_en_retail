import logging
import socket
from datetime import datetime, timezone
from pathlib import Path


def get_logger(
    script_name: str, log_dir: str = "artifacts/logs"
) -> logging.LoggerAdapter:
    """
    Crea un logger con:
    - timestamp
    - hostname en cada línea
    - salida a archivo + consola
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{script_name}_{timestamp}.log"

    base_logger = logging.getLogger(script_name)
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = False

    if not base_logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        base_logger.addHandler(file_handler)
        base_logger.addHandler(stream_handler)

    return logging.LoggerAdapter(base_logger, {"hostname": socket.gethostname()})
