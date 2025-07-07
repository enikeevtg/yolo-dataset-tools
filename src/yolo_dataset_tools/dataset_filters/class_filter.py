"""ClassFilter"""

from typing import List

from .base import BaseFilter
from ..models.yolo import DatasetInfo, ImageInfo


class ClassFilter(BaseFilter):
    def __init__(self, allowed_classes: List[str]) -> None:
        self.allowed_classes = allowed_classes

    def set_rules(self, dataset_info: DatasetInfo) -> None:
        extra = set(self.allowed_classes) - set(dataset_info.names)
        if extra:
            extra = ",".join(extra)
            raise NameError(f"{extra} is not present in the original dataset")

        orig_classes_dict = {
            name: class_id for class_id, name in enumerate(dataset_info.names)
        }
        allowed_classes_dict = {
            name: class_id
            for class_id, name in enumerate(self.allowed_classes)
        }
        self.rules = dict()
        self.allowed_orig_classes_ids = list()
        for name, class_id in allowed_classes_dict.items():
            self.allowed_orig_classes_ids.append(orig_classes_dict[name])
            self.rules[orig_classes_dict[name]] = class_id

    def transform_dataset_info(self, dataset_info: DatasetInfo) -> DatasetInfo:
        dataset_info.nc = len(self.allowed_classes)
        dataset_info.names = self.allowed_classes
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
            orig_class_id = int(line.split()[0])
            if orig_class_id in self.allowed_orig_classes_ids:
                new_annotation_lines.append(
                    f"{self.rules[orig_class_id]} "
                    f"{" ".join(line.split()[1:])}"
                )

        return new_annotation_lines
