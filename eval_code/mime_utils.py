from __future__ import annotations

import mimetypes
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


def detect_mime_type(path: str | Path) -> str:
    """Detect image MIME from file content before falling back to suffix."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            mime_type = IMAGE_FORMAT_TO_MIME.get(str(image.format or "").upper())
            if mime_type:
                return mime_type
    except (OSError, UnidentifiedImageError):
        pass
    mime_type, _ = mimetypes.guess_type(image_path.name)
    return mime_type or "application/octet-stream"
