"""Pure-NumPy 128x64 eye renderer and resilient SSD1309 SPI display driver."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EXPRESSIONS = ("neutral", "happy", "curious", "alert", "sleepy")
HEIGHT, WIDTH = 64, 128


def _rounded_rect(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, ry: float, rx: float, r: float) -> np.ndarray:
    dx = np.maximum(np.abs(xx - cx) - max(rx - r, 0.0), 0.0)
    dy = np.maximum(np.abs(yy - cy) - max(ry - r, 0.0), 0.0)
    return (dx * dx + dy * dy) <= (r * r)


class EyeRenderer:
    """Renders 1-bit 64x128 expressive eyes compatible with CyberDuck V2 visor orientation."""

    def render(
        self,
        expression: str = "neutral",
        blink_ratio: float = 0.0,
        gaze_x: float = 0.0,
        gaze_y: float = 0.0,
        rotate_180: bool = True,
    ) -> np.ndarray:
        if expression not in EXPRESSIONS:
            raise ValueError(f"Unsupported expression {expression!r}; expected one of {EXPRESSIONS}")

        blink = float(np.clip(blink_ratio, 0.0, 1.0))
        gx = float(np.clip(gaze_x, -1.0, 1.0)) * 9.0
        gy = float(np.clip(gaze_y, -1.0, 1.0)) * 6.0

        yy, xx = np.ogrid[:HEIGHT, :WIDTH]
        frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        open_scale = max(1.0 - blink, 0.05)

        for idx, base_cx in enumerate((38.0, 90.0)):
            cx = base_cx + gx
            cy = 32.0 + gy
            rx, ry = 16.0, 18.0 * open_scale
            if expression == "curious" and idx == 1:
                ry *= 1.18
                cy -= 2.5
            elif expression == "alert":
                rx, ry = 18.0, 21.0 * open_scale
            elif expression == "sleepy":
                ry *= 0.55
                cy += 3.0

            eye = _rounded_rect(yy, xx, cy, cx, max(ry, 1.5), rx, min(6.0, max(ry, 1.5)))
            if expression == "happy" and open_scale > 0.25:
                inner = _rounded_rect(yy, xx, cy + 7.0, cx, max(ry * 0.8, 1.0), rx * 0.78, 4.0)
                eye = eye & (~inner) & (yy <= cy + 4.0)
            elif expression == "sleepy" and open_scale > 0.2:
                eye = eye & (yy >= cy - 3.5 * open_scale)
            elif expression == "alert" and open_scale > 0.4:
                pupil = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 20.0
                eye = eye & (~pupil)
            frame[eye] = 1

        return frame[::-1, ::-1].copy() if rotate_180 else frame


class SSD1309Display:
    """Thin SPI OLED wrapper failing closed (returning False) on host/missing drivers."""

    def __init__(
        self,
        mock: bool = False,
        spi_port: int = 0,
        spi_device: int = 0,
        rotate_180: bool = True,
    ) -> None:
        self.mock = bool(mock)
        self.spi_port = int(spi_port)
        self.spi_device = int(spi_device)
        self.rotate_180 = bool(rotate_180)
        self.last_frame: Optional[np.ndarray] = None
        self._device = None
        self._warned_missing_driver = False

    def show(self, frame: np.ndarray) -> bool:
        arr = (np.asarray(frame) > 0).astype(np.uint8)
        if arr.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Expected frame shape {(HEIGHT, WIDTH)}, got {arr.shape}")
        self.last_frame = arr.copy()
        if self.mock:
            return True

        try:
            if self._device is None:
                from luma.core.interface.serial import spi
                from luma.oled.device import ssd1309

                serial = spi(port=self.spi_port, device=self.spi_device)
                self._device = ssd1309(serial, rotate=2 if self.rotate_180 else 0)
            from PIL import Image

            self._device.display(Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("1"))
            return True
        except Exception as exc:
            if not self._warned_missing_driver:
                logger.warning("SSD1309 SPI display unavailable (%s); continuing without OLED output", exc)
                self._warned_missing_driver = True
            return False
