"""ClassFilter"""

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple, Union

from .base import BaseFilter
from ..models.yolo import DatasetInfo, ImageInfo


@dataclass
class RelBoxSizeRanges:
    class_name: str
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

    def is_in_range(self, wn: float, hn: float, image_info: ImageInfo) -> bool:
        return (
            self.wn_min < wn < self.wn_max and self.hn_min < hn < self.hn_max
        )


# TODO pydantic
@dataclass
class AbsBoxSizeRanges:
    class_name: str
    w_px_min: int = 0
    w_px_max: int = 0
    h_px_min: int = 0
    h_px_max: int = 0

    def validate(self) -> bool:
        if not (
            self.w_px_min >= 0
            and self.h_px_min >= 0
            and self.w_px_min < self.w_px_max
            and self.h_px_min < self.h_px_max
        ):
            return False
        return True

    def is_in_range(self, wn: int, hn: int, image_info: ImageInfo) -> bool:
        wn_max = min(self.w_px_max, image_info.width)
        hn_max = min(self.h_px_max, image_info.height)
        if self.w_px_min < wn < wn_max and self.h_px_min < hn < hn_max:
            return True
        return False


BoxRule = Union[RelBoxSizeRanges, AbsBoxSizeRanges]


class BoxSizeFilter(BaseFilter):
    def __init__(self, size_ranges: List[BoxRule]) -> None:
        """
        :param size_ranges: box size limits dict with format:
        {"class_name": RelBoxSizeRanges(),...}
        """
        self.size_ranges = size_ranges

    def set_rules(self, dataset_info: DatasetInfo) -> None:
        class_names = [sr.class_name for sr in self.size_ranges]
        extra = set(class_names) - set(dataset_info.names)
        if extra:
            extra = ",".join(extra)
            raise NameError(f"{extra} is not present in the output subset")

        for sr in self.size_ranges:
            if not sr.validate():
                raise ValueError(
                    f"Incorrect range for class '{sr.class_name}'"
                )

        # build rules dict: {0: BoxSizeRanges(),...}
        class_dict = {name: idx for idx, name in enumerate(dataset_info.names)}
        self.rules: Dict[int, BoxRule] = dict()
        for size_range in self.size_ranges:
            if self.rules.get(class_dict[size_range.class_name], None):
                raise RuntimeError(
                    f"Duplicate description for '{size_range.class_name}'"
                )
            self.rules[class_dict[size_range.class_name]] = size_range
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
            x1, y1 = float(ann[1]), float(ann[2])
            x2, y2 = float(ann[3]), float(ann[4])
            x3, y3 = float(ann[5]), float(ann[6])
            if (x2 - x1) != 0.0 and -1.0 <= (y2 - y1) / (x2 - x1) <= 1.0:
                wn = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                hn = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            else:
                wn = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
                hn = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        else:
            """0        1   2   3  4  5"""
            """class_id xcn ycn wn hn (r)"""
            wn = float(ann[3])
            hn = float(ann[4])
        return wn, hn
