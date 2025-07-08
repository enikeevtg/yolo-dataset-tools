"""OrientationFilter"""

from typing import List

from .base import BaseFilter
from ..models.yolo import DatasetInfo, ImageInfo


PORTRAIT = "portrait"
LANDSCAPE = "landscape"


class OrientationFilter(BaseFilter):
    available_orientations = (PORTRAIT, LANDSCAPE)

    def __init__(self, orientation: str) -> None:
        """
        :param orientation: one of the two (PORTRAIT, LANDSCAPE)
        """

        if orientation not in self.available_orientations:
            raise ValueError(
                f"Orientation must be one of the two {self.available_orientations}"
            )
        self.orientation = orientation

    def set_rules(self, dataset_info: DatasetInfo) -> None:
        pass

    def transform_dataset_info(self, dataset_info: DatasetInfo) -> DatasetInfo:
        return dataset_info

    def apply(
        self,
        annotation_lines: List[str],
        image_info: ImageInfo,
    ) -> List[str]:
        """
        :param annotation_lines: [ "{cls} {xc or x1} {yc or y1} ...", ... ]
        """

        if (
            self.orientation == PORTRAIT
            and image_info.height < image_info.width
            or self.orientation == LANDSCAPE
            and image_info.height > image_info.width
        ):
            return None
        return annotation_lines
