"""ClassFilter"""

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple, Union

from .base import BaseFilter
from ..models.yolo import DatasetInfo, ImageInfo


@dataclass
class RelBoxSizeRanges:
    wn_min: float = 0.0
    wn_max: float = 1.0
    hn_min: float = 0.0
    hn_max: float = 1.0

    def validate(self) -> bool:
        if not (
            0.0 <= self.wn_min < 1.0
            and 0.0 < self.wn_max <= 1.0
            and 0.0 <= self.hn_min < 1.0
            and 0.0 < self.hn_max <= 1.0
            and self.wn_min < self.wn_max
            and self.hn_min < self.hn_max
        ):
            return False
        return True

    def is_in_range(self, wn: int, hn: int, image_info: ImageInfo) -> bool:
        if self.wn_min < wn < self.wn_max and self.hn_min < hn < self.hn_max:
            return True
        return False


# TODO pydantic
@dataclass
class AbsBoxSizeRanges:
    wn_min: int = 0
    wn_max: int = 0
    hn_min: int = 0
    hn_max: int = 0

    def validate(self) -> bool:
        if not (
            self.wn_min >= 0
            and self.hn_min >= 0
            and self.wn_min < self.wn_max
            and self.hn_min < self.hn_max
        ):
            return False
        return True

    def is_in_range(self, wn: int, hn: int, image_info: ImageInfo) -> bool:
        wn_max = min(self.wn_max, image_info.width)
        hn_max = min(self.hn_max, image_info.height)
        if self.wn_min < wn < wn_max and self.hn_min < hn < hn_max:
            return True
        return False


class BoxSizeFilter(BaseFilter):
    def __init__(
        self,
        size_ranges: Dict[str, Union[RelBoxSizeRanges, AbsBoxSizeRanges]],
    ) -> None:
        """
        :param size_ranges: box size limits dict with format:
        {"class_name": RelBoxSizeRanges(),...}
        """
        self.size_ranges = size_ranges

    def set_rules(self, dataset_info: DatasetInfo) -> None:
        extra = set(self.size_ranges.keys()) - set(dataset_info.names)
        if extra:
            extra = ",".join(extra)
            raise NameError(f"{extra} is not present in the output subset")

        for name, size_range in self.size_ranges.items():
            if not size_range.validate():
                raise ValueError(f"Incorrect range for class '{name}'")

        # build rules dict: {0: BoxSizeRanges(),...}
        class_dict = {name: idx for idx, name in enumerate(dataset_info.names)}
        self.rules = {
            class_dict[name]: ranges
            for name, ranges in self.size_ranges.items()
        }
        self.rule_classes_ids = self.rules.keys()

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

        new_annotation_lines = []
        for line in annotation_lines:
            split_line = line.split()
            class_id = int(split_line[0])
            if class_id not in self.rule_classes_ids:
                continue

            wn, hn = self._get_box_wn_hn(split_line)
            if isinstance(self.rules[class_id], AbsBoxSizeRanges):
                wn = int(wn * image_info.width)
                hn = int(hn * image_info.height)

            if self.rules[class_id].is_in_range(wn, hn, image_info):
                new_annotation_lines.append(line)

        return new_annotation_lines

    @staticmethod
    def _get_box_wn_hn(ann: List[str]) -> Tuple[float, float]:
        if len(ann) == 9:
            """0        1   2   3   4   5   6   7   8"""
            """class_id x1n y1n x2n y2n x3n y3n x4n y4n"""
            """Clockwise bypass and (x1 y1) is right up point"""
            wn = math.sqrt(
                (float(ann[3]) - float(ann[1])) ** 2
                + (float(ann[4]) - float(ann[2])) ** 2
            )
            hn = math.sqrt(
                (float(ann[5]) - float(ann[3])) ** 2
                + (float(ann[6]) - float(ann[4])) ** 2
            )
        else:
            """0        1  2  3  4  5"""
            """class_id xn yn wn hn (r)"""
            wn = float(ann[3])
            hn = float(ann[4])
        return wn, hn
