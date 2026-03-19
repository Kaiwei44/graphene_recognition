from .models import FieldOfViewCalibration, PixelScale


DEFAULT_GRAPHENE_FOV = FieldOfViewCalibration(
    fov_width_um=1247.52,
    fov_height_um=887.11,
)


def build_pixel_scale(
    image_width_px: int,
    image_height_px: int,
    calibration: FieldOfViewCalibration = DEFAULT_GRAPHENE_FOV,
) -> PixelScale:
    pixel_width_um = calibration.fov_width_um / image_width_px
    pixel_height_um = calibration.fov_height_um / image_height_px
    return PixelScale(
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
        pixel_area_um2=pixel_width_um * pixel_height_um,
    )
