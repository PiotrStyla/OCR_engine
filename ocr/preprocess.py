"""Przetwarzanie wstępne obrazu: wczytywanie, konwersja, korekta pochylenia (deskew).

Wszystkie funkcje operują na numpy arrays (BGR/RGB) z opencv, ale przyjmują też
ścieżki do plików. Zwracane obrazy są w formacie RGB (H, W, 3) uint8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageLike = Union[str, Path, np.ndarray]


def load_image(source: ImageLike) -> np.ndarray:
    """Wczytuje obraz z pliku lub przepuszcza array. Zwraca RGB uint8 (H,W,3)."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
            raise ValueError("Expected RGB uint8 array with shape (H, W, 3)")
        return arr
    else:
        path = str(source)
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Nie udało się wczytać obrazu: {path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _estimate_angle(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Szacuje kąt pochylenia tekstu poprzez analizę projekcji poziomej."""
    # binarny obraz (odwrotny, bo tekst jest ciemny na jasnym tle)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    best_angle = 0.0
    best_var = -1.0
    for angle in np.arange(-max_angle, max_angle + 0.5, 0.5):
        h, w = thr.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(thr, m, (w, h), flags=cv2.INTER_NEAREST)
        # projekcja pozioma: suma pikseli w wierszach
        projection = rotated.sum(axis=1)
        var = projection.var()
        if var > best_var:
            best_var = var
            best_angle = float(angle)
    return best_angle


def deskew(image: np.ndarray, max_angle: float = 5.0, *, return_transform=False):
    """Koryguje pochylenie tekstu (drobne rotacje)."""
    gray = to_grayscale(image)
    angle = _estimate_angle(gray, max_angle)
    if abs(angle) < 0.5:
        return (image, np.array([[1., 0., 0.], [0., 1., 0.]])) if return_transform else image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return (rotated, m) if return_transform else rotated


def crop_to_bbox(image: np.ndarray, bbox) -> np.ndarray:
    """Wycina fragment obrazu wg BBox (x1,y1,x2,y2). Akceptuje BBox lub tuple."""
    if hasattr(bbox, "to_tuple"):
        x1, y1, x2, y2 = bbox.to_tuple()
    else:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return image[y1:y2, x1:x2]


def to_pil_rgb(image: np.ndarray):
    """Konwertuje array RGB na PIL.Image (leniwy import PIL)."""
    from PIL import Image
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def parse_page_range(spec: str | None, n_pages: int) -> list[int]:
    """Parsuje zakres stron '1-3,5' (1-indeksowane) na listę indeksów 0-bazowanych."""
    if not spec:
        return list(range(n_pages))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            out.extend(range(start - 1, end))
        else:
            out.append(int(part) - 1)
    return [i for i in out if 0 <= i < n_pages]


def iter_pdf_pages(source, dpi: int = 300, pages: str | None = None):
    """Iteruje strony PDF jako obrazy RGB (H,W,3) uint8.

    `pages`: zakres 1-indeksowany, np. "1-3,5". None = wszystkie strony.
    Zwraca generator (numer_strony_1based, ndarray).
    """
    import fitz  # leniwy import (PyMuPDF)

    doc = fitz.open(str(source)) if not hasattr(source, "read") else fitz.open(stream=source.read(), filetype="pdf")
    try:
        idxs = parse_page_range(pages, doc.page_count)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in idxs:
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            yield i + 1, arr.copy()
    finally:
        doc.close()
