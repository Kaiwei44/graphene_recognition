from .area import measure_mask_area
from .calibration import DEFAULT_GRAPHENE_FOV, build_pixel_scale
from .models import AreaMeasurement, FieldOfViewCalibration


def measure_flake_area(
    flake,
    calibration: FieldOfViewCalibration = DEFAULT_GRAPHENE_FOV,
) -> AreaMeasurement:
    image_height_px, image_width_px = flake.mask.shape
    pixel_scale = build_pixel_scale(
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        calibration=calibration,
    )
    return measure_mask_area(flake.mask, pixel_scale)


def attach_measurements_to_flake(
    flake,
    calibration: FieldOfViewCalibration = DEFAULT_GRAPHENE_FOV,
):
    flake.measurements = measure_flake_area(flake, calibration)
    return flake
