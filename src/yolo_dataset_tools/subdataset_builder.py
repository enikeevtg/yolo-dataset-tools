"""DatasetSubsetBulder"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Union
import yaml

from .dataset_filters.base import BaseFilter
from .filemanager import filemanager as fm
from .models.yolo import DatasetInfo, ImageInfo


class SubDatasetBuilder:
    """
    ### Dataset subset building

    Original dataset schema:

    dataset/
        ├── images/
        ├── labels/
        └── classes.txt or data.yaml
    """

    def __init__(self, dataset_path: str | Path, *args) -> None:
        """
        :param dataset_path: аболютный путь к директории, содержащей папки images и labels с изображениями и аннотациями соответственно, а также файл classes.txt или data.yaml
        """

        self.dataset_path = fm.resolve_path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        self.dataset_info = self._load_dataset_info()
        self.subset_info = self.dataset_info
        self.filters: List[BaseFilter] = []

    def add_filter(self, filter: BaseFilter):
        filter.set_rules(self.subset_info)
        self.subset_info = filter.transform_dataset_info(self.subset_info)
        self.filters.append(filter)

    def build_subset(self, subset_path: Union[str, Path]):
        """
        Dataset subset build running
        :param subset_path: subset path. Warning: subset directory will be cleaned!
        """

        if not self.filters:
            raise RuntimeError("Filters list is empty")

        subset_path = fm.resolve_path(subset_path)
        subset_images_path = subset_path / "images"
        subset_labels_path = subset_path / "labels"
        for dir in (subset_path, subset_images_path, subset_labels_path):
            fm.remove_dir(dir)
            fm.create_dir(dir)

        count = 0
        for image_filename in tqdm(
            os.listdir(self.images_path), desc="Processed", leave=True
        ):
            with Image.open(self.images_path / image_filename) as img:
                image = ImageInfo(width=img.size[0], height=img.size[1])

            label_filename = Path(image_filename).stem + ".txt"
            annotations = self._load_annotations(
                self.labels_path / label_filename
            )
            for filter in self.filters:
                if not annotations:
                    break
                annotations = filter.apply(annotations, image)

            if annotations:
                count += 1
                self._dump_image_annotations(
                    annotations_path=subset_labels_path / label_filename,
                    annotations=annotations,
                )
                fm.copy_file(
                    src=self.images_path / image_filename,
                    dst=subset_images_path / image_filename,
                )

        print(f"{count} files added to subset")
        self._transform_dataset_info()
        self._dump_dataset_metadata(subset_path)

    def _load_dataset_info(self) -> DatasetInfo:
        data_yaml_path = self.dataset_path / "data.yaml"
        classes_txt_path = self.dataset_path / "classes.txt"
        if not fm.is_file(data_yaml_path) and not fm.is_file(classes_txt_path):
            raise FileNotFoundError(f"Dataset metadata files not found")

        if not fm.is_file(data_yaml_path):
            return self.convert_classes_txt_to_data_yaml(
                classes_txt_path, data_yaml_path, save=False
            )

        with open(data_yaml_path, "r", encoding="utf-8") as fp:
            content = yaml.safe_load(fp)

        return DatasetInfo.from_dict(content)

    def _load_annotations(self, annotations_path: Path) -> List[str]:
        if fm.is_file(annotations_path):
            with open(annotations_path, "r") as fp:
                return [line.strip() for line in fp.readlines()]

    def _dump_image_annotations(
        self,
        annotations_path: str | Path,
        annotations: List[str],
    ) -> None:
        with open(annotations_path, "w") as fp:
            fp.write("\n".join(annotations))

    def _transform_dataset_info(self):
        self.subset_info.train = "images"
        self.subset_info.val = ""
        self.subset_info.test = ""

    def _dump_dataset_metadata(self, subset_path: Path) -> None:
        with open(subset_path / "data.yaml", "w") as fp:
            fp.write(str(self.subset_info))
            print(f"data.yaml created: {subset_path / "data.yaml"}")
        with open(subset_path / "classes.txt", "w") as fp:
            fp.write("\n".join(self.subset_info.names))
            print(f"classes.txt created: {subset_path / "classes.txt"}")

    @staticmethod
    def convert_classes_txt_to_data_yaml(
        classes_txt_path: str | Path,
        data_yaml_path: Union[str | Path] = None,
        save: bool = True,
    ) -> DatasetInfo:
        with open(classes_txt_path, "r", encoding="utf-8") as fp:
            classes = [line.strip() for line in fp.readlines()]
            dataset_info = DatasetInfo(
                train="images",
                val="",
                test="",
                nc=len(classes),
                names=classes,
            )

        if save:
            if not data_yaml_path:
                data_yaml_path = "data.yaml"
            with open(data_yaml_path, "w") as fp:
                fp.write(str(dataset_info))

        return dataset_info
