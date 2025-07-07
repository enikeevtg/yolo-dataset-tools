# YOLO dataset tools

A lightweight set of tools for working with YOLO datasets.


## ✨ Features

<!-- - 🔄 Convert from COCO, Pascal VOC, and LabelMe to YOLO format -->
- 🔄 Convert from COCO to YOLO format
- 🔍 Validate dataset integrity (image-label matching, missing files, duplicates)
- 🖼️ Visualize annotations for quick inspection
- 🧪 Generate train/val/test splits
<!-- - 📦 Lightweight and easy to integrate into ML pipelines -->


## 📦 Installation

From PyPI:
```bash
pip install yolo-dataset-tools
```

Or directly from GitHub:

```bash
pip install git+https://github.com/enikeevtg/yolo-dataset-tools.git
pip install -r yolo_datase_tools/requirements.txt
```


## 🚀 Quick Start

<!-- CLI example:

```bash
python coco2yolo_converter.py \
    --coco_path ./annotations/instances_train.json \
    --images_dir ./images \
    --output_dir ./output
```

Python example:

```python
from coco2yolo_converter import CocoToYoloConverter

converter = CocoToYoloConverter(
    coco_path='annotations.json',
    images_dir='images',
    output_dir='output'
)
converter.convert()
``` -->


<!-- ## 🛠️ CLI Usage

yolo-tools convert --format coco2yolo --input datasets/coco/ --output datasets/yolo/ -->

## 📌 Planned Features

+ dataset filtering
+ script usage with yaml configuration
+ Bounding box validation
+ CLI wrapper / entrypoint


## 📄 License
This project is licensed under the MIT License.


<p align="center"> Made with ❤️ for data scientists and ML engineers </p>
