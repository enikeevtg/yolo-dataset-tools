"""BaseFilter"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..models.yolo import DatasetInfo, ImageInfo


class BaseFilter(ABC):
    """if need object with last call parameters comment raise line"""

    _instances = dict()

    def __new__(cls, *args, **kwargs):
        if cls in cls._instances:
            raise RuntimeError(f"Instance of {cls.__name__} already exists")
            # return cls._instances[cls]  # object with last call parameters
        instance = super().__new__(cls)
        cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def set_rules(self, dataset_info: DatasetInfo) -> None:
        pass

    @abstractmethod
    def transform_dataset_info(self, dataset_info: DatasetInfo) -> DatasetInfo:
        pass

    @abstractmethod
    def apply(self, annotation: List[str], image_info: ImageInfo) -> List[str]:
        pass


# Singleton
# from abc import ABCMeta

# class MetaFilterSingleton(ABCMeta):
#     """if need object with first call parameters comment raise line"""

#     _instances = dict()

#     def __call__(cls, *args, **kwargs):
#         if cls in cls._instances:
#             raise RuntimeError(f"Instance of {cls.__name__} already exists")
#             # return cls._instances[cls]
#         instance = super().__call__(*args, **kwargs)
#         cls._instances[cls] = instance
#         return cls._instances[cls]


# class BaseFilter(ABC, metaclass=MetaFilterSingleton):
#     ...
