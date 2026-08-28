from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class CameraFrame:
    timestamp: datetime
    width: int
    height: int
    rgb: bytes


class MockCamera:
    """Synthetic RGB camera with moving shapes and a UTC timestamp overlay."""

    def __init__(self, width: int = 320, height: int = 240, jpeg_quality: int = 80) -> None:
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self._frame_index = 0
        self._lock = threading.Lock()

    def capture(self) -> CameraFrame:
        with self._lock:
            frame_index = self._frame_index
            self._frame_index += 1

        timestamp = datetime.now(UTC)
        image = Image.new("RGB", (self.width, self.height), (232, 238, 244))
        draw = ImageDraw.Draw(image)

        grid = 32
        for x in range(0, self.width, grid):
            draw.line((x, 0, x, self.height), fill=(205, 215, 225), width=1)
        for y in range(0, self.height, grid):
            draw.line((0, y, self.width, y), fill=(205, 215, 225), width=1)

        phase = frame_index / 12.0
        center_x = int(self.width / 2 + math.sin(phase) * self.width * 0.28)
        center_y = int(self.height / 2 + math.cos(phase * 0.7) * self.height * 0.22)
        radius = max(12, min(self.width, self.height) // 12)
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=(14, 116, 144),
            outline=(8, 75, 94),
            width=3,
        )
        draw.rectangle((8, 8, self.width - 8, 38), fill=(15, 23, 42))
        draw.text(
            (16, 17),
            f"MOCK CAM  {timestamp.isoformat(timespec='milliseconds')}",
            fill=(241, 245, 249),
        )
        return CameraFrame(
            timestamp=timestamp,
            width=self.width,
            height=self.height,
            rgb=image.tobytes(),
        )

    def jpeg(self, frame: CameraFrame | None = None) -> bytes:
        frame = frame or self.capture()
        image = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=self.jpeg_quality)
        return output.getvalue()
