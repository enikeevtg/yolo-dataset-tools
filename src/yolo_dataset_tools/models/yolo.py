from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DatasetInfo:
    train: str
    val: str
    test: str
    nc: int
    names: List[str]

    def __repr__(self):
        return (
            f"train: {self.train}\n"
            f"val: {self.val}\n"
            f"test: {self.test}\n\n"
            f"nc: {self.nc}\n"
            f"names: {self.names}"
        )

    @staticmethod
    def from_dict(data: Dict):
        """:param data_yaml: serialized yaml content"""

        return DatasetInfo(
            train=data.get("train", ""),
            val=data.get("val", ""),
            test=data.get("test", ""),
            nc=len(data["names"]),
            names=data["names"],
        )


@dataclass
class ImageInfo:
    width: int
    height: int
