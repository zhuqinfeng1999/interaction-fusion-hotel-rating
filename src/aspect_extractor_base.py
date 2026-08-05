from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


ASPECTS: List[str] = [
    "service",
    "cleanliness",
    "location",
    "rooms",
    "value",
    "sleep_quality",
]


class AspectExtractor(ABC):
    aspects = ASPECTS

    @abstractmethod
    def extract_one(self, text: str) -> Dict[str, Dict[str, object]]:
        raise NotImplementedError

    @staticmethod
    def empty_output() -> Dict[str, Dict[str, object]]:
        return {
            aspect: {"mentioned": 0, "sentiment": 0, "evidence": ""}
            for aspect in ASPECTS
        }
