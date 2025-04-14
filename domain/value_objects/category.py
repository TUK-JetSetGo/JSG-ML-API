"""
카테고리 값 객체 모듈
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class CategoryType(Enum):
    """관광지 카테고리 유형"""

    NATURE = auto()  # 자연
    CULTURE = auto()  # 문화
    HISTORY = auto()  # 역사
    FOOD = auto()  # 음식
    SHOPPING = auto()  # 쇼핑
    ENTERTAINMENT = auto()  # 엔터테인먼트
    RELAXATION = auto()  # 휴양
    ADVENTURE = auto()  # 모험
    EDUCATION = auto()  # 교육
    OTHER = auto()  # 기타


@dataclass(frozen=True)
class Category:
    """관광지 카테고리를 나타내는 값 객체"""

    name: str
    type: CategoryType = CategoryType.OTHER

    @classmethod
    def from_string(cls, category_str: str) -> "Category":
        """문자열에서 카테고리 객체 생성"""
        # 카테고리 문자열을 정규화 (소문자로 변환, 공백 제거)
        normalized = category_str.lower().strip()

        # 카테고리 유형 매핑
        type_mapping = {
            "nature": CategoryType.NATURE,
            "natural": CategoryType.NATURE,
            "mountain": CategoryType.NATURE,
            "beach": CategoryType.NATURE,
            "park": CategoryType.NATURE,
            "garden": CategoryType.NATURE,
            "forest": CategoryType.NATURE,
            "lake": CategoryType.NATURE,
            "river": CategoryType.NATURE,
            "waterfall": CategoryType.NATURE,
            "island": CategoryType.NATURE,
            "culture": CategoryType.CULTURE,
            "cultural": CategoryType.CULTURE,
            "art": CategoryType.CULTURE,
            "museum": CategoryType.CULTURE,
            "gallery": CategoryType.CULTURE,
            "theater": CategoryType.CULTURE,
            "concert": CategoryType.CULTURE,
            "festival": CategoryType.CULTURE,
            "history": CategoryType.HISTORY,
            "historical": CategoryType.HISTORY,
            "heritage": CategoryType.HISTORY,
            "monument": CategoryType.HISTORY,
            "castle": CategoryType.HISTORY,
            "palace": CategoryType.HISTORY,
            "temple": CategoryType.HISTORY,
            "shrine": CategoryType.HISTORY,
            "ruins": CategoryType.HISTORY,
            "food": CategoryType.FOOD,
            "restaurant": CategoryType.FOOD,
            "cafe": CategoryType.FOOD,
            "bar": CategoryType.FOOD,
            "dining": CategoryType.FOOD,
            "cuisine": CategoryType.FOOD,
            "gourmet": CategoryType.FOOD,
            "shopping": CategoryType.SHOPPING,
            "shop": CategoryType.SHOPPING,
            "mall": CategoryType.SHOPPING,
            "market": CategoryType.SHOPPING,
            "store": CategoryType.SHOPPING,
            "outlet": CategoryType.SHOPPING,
            "entertainment": CategoryType.ENTERTAINMENT,
            "amusement": CategoryType.ENTERTAINMENT,
            "theme park": CategoryType.ENTERTAINMENT,
            "zoo": CategoryType.ENTERTAINMENT,
            "aquarium": CategoryType.ENTERTAINMENT,
            "cinema": CategoryType.ENTERTAINMENT,
            "nightlife": CategoryType.ENTERTAINMENT,
            "relaxation": CategoryType.RELAXATION,
            "spa": CategoryType.RELAXATION,
            "hot spring": CategoryType.RELAXATION,
            "resort": CategoryType.RELAXATION,
            "wellness": CategoryType.RELAXATION,
            "adventure": CategoryType.ADVENTURE,
            "hiking": CategoryType.ADVENTURE,
            "trekking": CategoryType.ADVENTURE,
            "climbing": CategoryType.ADVENTURE,
            "diving": CategoryType.ADVENTURE,
            "surfing": CategoryType.ADVENTURE,
            "skiing": CategoryType.ADVENTURE,
            "rafting": CategoryType.ADVENTURE,
            "education": CategoryType.EDUCATION,
            "science": CategoryType.EDUCATION,
            "library": CategoryType.EDUCATION,
            "observatory": CategoryType.EDUCATION,
        }

        # 카테고리 유형 결정
        category_type = CategoryType.OTHER
        for key, value in type_mapping.items():
            if key in normalized:
                category_type = value
                break

        return cls(name=category_str, type=category_type)


@dataclass
class CategorySet:
    """관광지의 카테고리 집합을 나타내는 클래스"""

    categories: Set[Category] = field(default_factory=set)

    def add(self, category: Category) -> None:
        """카테고리 추가"""
        self.categories.add(category)

    def add_from_string(self, category_str: str) -> None:
        """문자열에서 카테고리 생성 후 추가"""
        self.add(Category.from_string(category_str))

    def add_from_list(self, category_list: List[str]) -> None:
        """문자열 리스트에서 카테고리 생성 후 추가"""
        for category_str in category_list:
            self.add_from_string(category_str)

    def has_type(self, category_type: CategoryType) -> bool:
        """특정 유형의 카테고리가 있는지 확인"""
        return any(category.type == category_type for category in self.categories)

    def match_score(self, other: "CategorySet") -> float:
        """다른 카테고리 집합과의 일치 점수 계산 (0.0 ~ 1.0)"""
        if not self.categories or not other.categories:
            return 0.0

        # 카테고리 유형 집합 생성
        self_types = {category.type for category in self.categories}
        other_types = {category.type for category in other.categories}

        # 교집합 크기 / 합집합 크기 (자카드 유사도)
        intersection = len(self_types.intersection(other_types))
        union = len(self_types.union(other_types))

        return intersection / union if union > 0 else 0.0

    def to_list(self) -> List[str]:
        """카테고리 이름 리스트 반환"""
        return [category.name for category in self.categories]

    def __len__(self) -> int:
        return len(self.categories)
