from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ImageType(Enum):
    BIAS = "BIAS"
    DARK = "DARK"
    FLAT = "FLAT"
    SCIENCE = "SCIENCE"


@dataclass
class FitsHeaderEntry:
    key: str
    value: Optional[str] = None
    comment: Optional[str] = None
