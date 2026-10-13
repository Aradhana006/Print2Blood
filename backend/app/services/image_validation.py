from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}
ALLOWED_MIME = {
    "image/bmp",
    "image/x-ms-bmp",
    "image/png",
    "image/jpeg",
    "application/octet-stream",  # some browsers send this for BMP
}


def validate_and_save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> dict:
    filename = Path(upload.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: BMP, PNG, JPG, JPEG.",
        )

    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {max_bytes // (1024 * 1024)} MB.",
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    try:
        with Image.open(dest) as img:
            img.verify()
        with Image.open(dest) as img:
            width, height = img.size
            mode = img.mode
    except (UnidentifiedImageError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Unreadable image: {exc}") from exc

    if width < 16 or height < 16:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Image dimensions are too small.")

    digest = hashlib.sha256(content).hexdigest()
    return {
        "filename": filename,
        "path": dest,
        "width": width,
        "height": height,
        "mode": mode,
        "image_hash": digest,
        "size_bytes": len(content),
    }
