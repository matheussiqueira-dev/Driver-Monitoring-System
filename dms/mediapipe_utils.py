from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


def ensure_model_asset(model_path: str, url: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Falha ao baixar o modelo do MediaPipe: {url}. "
            f"Baixe manualmente e salve em {path}."
        ) from exc
    return str(path)
