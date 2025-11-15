"""Shared helper utilities for the Shiny attention explorer."""

from io import BytesIO
import base64

import matplotlib.pyplot as plt
import numpy as np


def positional_encoding(position: int, d_model: int = 768) -> np.ndarray:
    """Sinusoidal positional encodings to mimic transformer inputs."""
    pe = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe


def array_to_base64_img(array: np.ndarray, cmap: str = "Blues", height: float = 0.22) -> str:
    """Encode a 1D numpy array as a small PNG strip for inline HTML usage."""
    plt.figure(figsize=(3, height))
    plt.imshow(array[np.newaxis, :], cmap=cmap, aspect="auto")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


__all__ = ["positional_encoding", "array_to_base64_img"]
