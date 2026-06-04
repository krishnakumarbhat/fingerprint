from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch


class BmpFormatError(RuntimeError):
    """Raised when a BMP file is not one of the SOCOFing-supported formats."""


def _read_header(data: bytes) -> tuple[int, int, int, int, int, int, int]:
    if data[:2] != b"BM":
        raise BmpFormatError("Only BMP files are supported.")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    width, height, planes, bits_per_pixel, compression = struct.unpack_from(
        "<iiHHI",
        data,
        18,
    )
    if planes != 1:
        raise BmpFormatError("Unsupported BMP plane count.")
    return pixel_offset, dib_size, width, height, bits_per_pixel, compression, abs(height)


def _row_stride(width: int, bits_per_pixel: int) -> int:
    return ((width * bits_per_pixel + 31) // 32) * 4


def _mask_to_channel(values: np.ndarray, mask: int) -> np.ndarray:
    if mask == 0:
        return np.zeros_like(values, dtype=np.uint8)
    shift = (mask & -mask).bit_length() - 1
    channel = (values & mask) >> shift
    channel_max = mask >> shift
    if channel_max == 255:
        return channel.astype(np.uint8)
    scaled = np.round(channel.astype(np.float32) * (255.0 / float(channel_max)))
    return scaled.astype(np.uint8)


def _decode_paletted(data: bytes, width: int, height: int, pixel_offset: int, dib_size: int) -> np.ndarray:
    palette_bytes = pixel_offset - 14 - dib_size
    palette_entries = palette_bytes // 4
    palette_offset = 14 + dib_size
    palette = np.frombuffer(
        data,
        dtype=np.uint8,
        count=palette_entries * 4,
        offset=palette_offset,
    ).reshape(palette_entries, 4)
    grayscale_palette = np.round(palette[:, :3].mean(axis=1)).astype(np.uint8)
    stride = _row_stride(width, 8)
    rows = np.frombuffer(
        data,
        dtype=np.uint8,
        count=stride * height,
        offset=pixel_offset,
    ).reshape(height, stride)
    indices = rows[:, :width]
    return grayscale_palette[indices]


def _decode_32bit(data: bytes, width: int, height: int, pixel_offset: int, dib_size: int) -> np.ndarray:
    if dib_size >= 56:
        red_mask, green_mask, blue_mask, alpha_mask = struct.unpack_from("<IIII", data, 54)
    else:
        red_mask, green_mask, blue_mask, alpha_mask = 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000
    pixels = np.frombuffer(
        data,
        dtype="<u4",
        count=width * height,
        offset=pixel_offset,
    ).reshape(height, width)
    red = _mask_to_channel(pixels, red_mask)
    green = _mask_to_channel(pixels, green_mask)
    blue = _mask_to_channel(pixels, blue_mask)
    if alpha_mask:
        alpha = _mask_to_channel(pixels, alpha_mask).astype(np.float32) / 255.0
    else:
        alpha = np.ones((height, width), dtype=np.float32)
    grayscale = (0.299 * red + 0.587 * green + 0.114 * blue) * alpha
    return np.round(grayscale).clip(0, 255).astype(np.uint8)


def _decode_24bit(data: bytes, width: int, height: int, pixel_offset: int) -> np.ndarray:
    stride = _row_stride(width, 24)
    rows = np.frombuffer(
        data,
        dtype=np.uint8,
        count=stride * height,
        offset=pixel_offset,
    ).reshape(height, stride)
    pixels = rows[:, : width * 3].reshape(height, width, 3)
    blue = pixels[:, :, 0].astype(np.float32)
    green = pixels[:, :, 1].astype(np.float32)
    red = pixels[:, :, 2].astype(np.float32)
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.round(grayscale).clip(0, 255).astype(np.uint8)


def load_socofing_bmp(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    pixel_offset, dib_size, width, height, bits_per_pixel, compression, height_abs = _read_header(data)
    if bits_per_pixel == 8 and compression == 0:
        image = _decode_paletted(data, width, height_abs, pixel_offset, dib_size)
    elif bits_per_pixel == 24 and compression == 0:
        image = _decode_24bit(data, width, height_abs, pixel_offset)
    elif bits_per_pixel == 32 and compression in {0, 3}:
        image = _decode_32bit(data, width, height_abs, pixel_offset, dib_size)
    else:
        raise BmpFormatError(
            f"Unsupported BMP encoding: bpp={bits_per_pixel}, compression={compression}, path={path}"
        )
    if height > 0:
        image = np.flipud(image)
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).unsqueeze(0).float() / 255.0