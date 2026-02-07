"""logging_config.py

Utilidades para configurar logging consistente en los scripts del pipeline.

- Uso de logging.basicConfig(...)
- Niveles: DEBUG/INFO/WARNING/ERROR/CRITICAL
- Formato con timestamp y contexto (módulo / hostname)
- LoggerAdapter para agregar hostname
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path


def setup_logging(
    log_file: Path,
    level: int = logging.INFO,
    filemode: str = "w",
) -> logging.LoggerAdapter:
    """Configura logging para escribir a un archivo.

    Parameters
    ----------
    log_file : Path
        Ruta del archivo .log donde se guardarán los eventos.
    level : int, optional
        Nivel mínimo de severidad a registrar (default: logging.INFO).
    filemode : str, optional
        Modo del archivo (default: "w" para overwrite; usar "a" para append).

    Returns
    -------
    logging.LoggerAdapter
        Logger con contexto adicional (hostname).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(log_file),
        level=level,
        filemode=filemode,
        format="%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    base_logger = logging.getLogger(__name__)
    return logging.LoggerAdapter(base_logger, {"hostname": socket.gethostname()})