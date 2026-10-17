from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from .errors import ModelDownloadError

DOWNLOAD_TIMEOUT_S = 60
ALLOWED_MODEL_HOSTS = frozenset({"storage.googleapis.com"})


def ensure_model_asset(model_path: str, url: str) -> str:
    """Return a local model path, downloading it once if missing.

    The download is written to a temporary file and moved into place only when
    complete, so an interrupted download never leaves a corrupt model behind.
    """
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_MODEL_HOSTS:
        raise ModelDownloadError(f"URL de modelo nao permitida: {url}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp, urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310 - https + host allowlist above
            while chunk := response.read(1 << 16):
                tmp.write(chunk)
        os.replace(tmp_name, path)
    except Exception as exc:  # pragma: no cover - network dependent
        Path(tmp_name).unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Falha ao baixar o modelo do MediaPipe: {url}. Baixe manualmente e salve em {path}."
        ) from exc
    return str(path)
