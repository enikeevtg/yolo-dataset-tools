import math
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from ..models.coco import *
from ..filemanager import filemanager as fm


class Coco2YoloConverter:
    YOLO_FORMATS = ["yolo", "yolo-obb"]

    def __init__(
        self,
        yolo_dataset_type: str,
        yolo_dataset_path: str | Path,
        input_json_path: str | Path,
        input_images_path: str | Path,
        output_labels_path: str | Path,
        output_images_path: str | Path,
    ):

        if yolo_dataset_type not in self.YOLO_FORMATS:
            raise TypeError(
                f"Supported yolo dataset formats: {self.YOLO_FORMATS}"
            )

        self.yolo_dataset_type = yolo_dataset_type
        self.yolo_dataset_path = Path(yolo_dataset_path)
        self.input_json_path = Path(input_json_path)
        self.input_images_path = Path(input_images_path)
        self.output_labels_path = Path(output_labels_path)
        self.output_images_path = Path(output_images_path)
        fm.create_dir(output_images_path, exist_ok=True)
        fm.create_dir(output_labels_path, exist_ok=True)

    def run(self):
        classes, annotations, images = self._load_coco_dataset()
        yolo_annotations = self._convert(annotations, images)
        self._save_yolo_dataset(classes, images, yolo_annotations)

    def _load_coco_dataset(self) -> Tuple[
        Dict[int, str],
        List[Annotation],
        Dict[int, Image],
    ]:
        with open(self.input_json_path, "r") as fp:
            coco_data = COCO.model_validate_json(fp.read())
        classes: Dict[int, Class] = {}
        for _class in coco_data.classes:
            classes[_class.id] = _class.name
        images: Dict[int, Image] = {}
        for image in coco_data.images:
            images[image.id] = image
        return classes, coco_data.annotations, images

    def _convert(
        self,
        annotations: List[Annotation],
        images: Dict[int, Image],
    ) -> Dict[int, List[str]]:
        yolo_annotations: Dict[int, List[str]] = {}
        for annotation in tqdm(
            iterable=annotations,
            desc="Converting",
            leave=True,
            position=0,
        ):
            yolo_bbox = self._get_bbox(
                annotation.bbox,
                images[annotation.image_id].width,
                images[annotation.image_id].height,
            )
            yolo_annotations.setdefault(annotation.image_id, []).append(
                f"{annotation.class_id - 1} {" ".join(map(str, yolo_bbox))}"
            )
        return yolo_annotations

    def _get_bbox(
        self,
        coco_bbox: List[float],
        image_w: float,
        image_h: float,
        box_rotation_deg: float,
    ) -> List[float]:
        x0, y0, box_w, box_h = coco_bbox
        if self.yolo_dataset_type == "yolo":
            bbox = [
                (x0 + box_w / 2) / image_w,
                (y0 + box_h / 2) / image_h,
                box_w / image_w,
                box_h / image_h,
            ]
        elif self.yolo_dataset_type == "yolo-obb":
            cx = x0 + box_w / 2
            cy = y0 + box_h / 2
            corners = [
                (-box_w / 2, -box_h / 2),  # top-left
                (box_w / 2, -box_h / 2),  # top-right
                (box_w / 2, box_h / 2),  # bottom-right
                (-box_w / 2, box_h / 2),  # bottom-left
            ]
            yolo_bbox = []
            box_rotation_rad = math.radians(box_rotation_deg)
            cos_a = math.cos(box_rotation_rad)
            sin_a = math.sin(box_rotation_rad)
            for x, y in corners:
                x_rot = cos_a * x - sin_a * y + cx
                y_rot = sin_a * x + cos_a * y + cy
                x_norm = x_rot / image_w
                y_norm = y_rot / image_h
                yolo_bbox.extend([x_norm, y_norm])
        return yolo_bbox

    def _save_yolo_dataset(
        self,
        classes: Dict[int, str],
        images: Dict[int, Image],
        yolo_annotations: Dict[int, List[str]],
    ) -> None:
        # create classes.txt
        with open(f"{self.yolo_dataset_path}/classes.txt", "w") as fp:
            for i in range(len(classes)):
                fp.write(f"{classes[i + 1]}\n")

        count = 0
        for image_id, annotation in tqdm(
            iterable=yolo_annotations.items(),
            desc="Saving",
            leave=True,
            position=0,
        ):
            count += 1
            # copy image
            image_filename = Path(images[image_id].file_name)
            fm.copy_file(
                self.input_images_path / image_filename,
                self.output_images_path / image_filename,
            )
            # create label text file
            label_filename = f"{image_filename.stem}.txt"
            with open(self.output_labels_path / label_filename, "w") as fp:
                for line in annotation:
                    fp.writelines(f"{line}\n")
