from pydantic import BaseModel, Field
from typing import List


class License(BaseModel):
    name: str = ""
    id: int = 0
    url: str = ""


class Info(BaseModel):
    contributor: str = ""
    date_created: str = ""
    description: str = ""
    url: str = ""
    version: str = ""
    year: str = ""


class Class(BaseModel):
    id: int = 1
    name: str = ""
    superclass: str = ""


class Image(BaseModel):
    id: int = 0
    width: int = 0
    height: int = 0
    file_name: str = ""
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: int = 0


class AnnotationAttrbutes(BaseModel):
    occluded: bool = False
    rotation: float = 0.0


class Annotation(BaseModel):
    id: int = 0
    image_id: int = 0
    class_id: int = 0
    segmentation: List = Field(default_factory=list)
    area: float = 0.0
    bbox: List = Field(default_factory=list)
    # bbox = [x0, y0, width, height]
    iscrowd: int = (0,)
    attributes: AnnotationAttrbutes = AnnotationAttrbutes()


class COCO(BaseModel):
    licenses: List[License] = Field(default_factory=list)
    info: Info = Info()
    classes: List[Class] = Field(default_factory=list)
    images: List[Image] = Field(default_factory=list)
    annotations: List[Annotation] = Field(default_factory=list)
