import numpy as np

from .models import AreaMeasurement, PixelScale


def count_mask_pixels(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def measure_mask_area(mask: np.ndarray, pixel_scale: PixelScale) -> AreaMeasurement:
    area_px = count_mask_pixels(mask)
    return AreaMeasurement(
        area_px=area_px,
        area_um2=area_px * pixel_scale.pixel_area_um2,
        pixel_scale=pixel_scale,
    )
