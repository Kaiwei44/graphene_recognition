from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FieldOfViewCalibration:
    fov_width_um: float
    fov_height_um: float


@dataclass(slots=True, frozen=True)  
class PixelScale:
    image_width_px: int
    image_height_px: int
    pixel_width_um: float
    pixel_height_um: float
    pixel_area_um2: float

    def to_dict(self) -> dict:
        return {
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "pixel_width_um": self.pixel_width_um,
            "pixel_height_um": self.pixel_height_um,
            "pixel_area_um2": self.pixel_area_um2,
        }


@dataclass(slots=True, frozen=True)
class AreaMeasurement:
    area_px: int
    area_um2: float
    pixel_scale: PixelScale

    def to_dict(self) -> dict:
        return {
            "area_px": self.area_px,
            "area_um2": self.area_um2,
            **self.pixel_scale.to_dict(),
        }
