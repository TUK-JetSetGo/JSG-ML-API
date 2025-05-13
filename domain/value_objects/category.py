"""
카테고리 값 객체 모듈
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class CategoryType(Enum):
    NATURE = "자연"
    CULTURE = "문화"
    HISTORY = "역사"
    FOOD = "음식"
    SHOPPING = "쇼핑"
    ENTERTAINMENT = "엔터테인먼트"
    RELAXATION = "휴식"
    ADVENTURE = "모험"
    OTHER = "기타"  # 추가


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
            "불교": CategoryType.HISTORY,  # 종교적·역사적 요소로 판단
            "릉": CategoryType.HISTORY,  # 역사적 유산
            "묘": CategoryType.HISTORY,  # 유적/역사
            "총": CategoryType.HISTORY,  # 의미가 다소 모호하나 역사적 요소로 판단
            "테마파크": CategoryType.ENTERTAINMENT,
            "폭포": CategoryType.NATURE,
            "휴양림": CategoryType.NATURE,
            "산림욕장": CategoryType.NATURE,
            "섬": CategoryType.NATURE,
            "계곡": CategoryType.NATURE,
            "자연명소": CategoryType.NATURE,
            "기념물": CategoryType.HISTORY,
            "레일바이크": CategoryType.ADVENTURE,
            "여행": CategoryType.ENTERTAINMENT,  # 여행 전반은 즐거움(엔터테인먼트) 관점으로 분류
            "명소": CategoryType.ENTERTAINMENT,
            "레저": CategoryType.ENTERTAINMENT,
            "테마": CategoryType.ENTERTAINMENT,
            "식물원": CategoryType.NATURE,
            "수목원": CategoryType.NATURE,
            "워터파크": CategoryType.ENTERTAINMENT,
            "지역명소": CategoryType.ENTERTAINMENT,
            "드라이브": CategoryType.ENTERTAINMENT,  # 즐기는 드라이브 코스 등
            "아쿠아리움": CategoryType.ENTERTAINMENT,
            "오름": CategoryType.NATURE,  # 제주 오름 등 자연형태
            "박물관": CategoryType.CULTURE,
            "자연": CategoryType.NATURE,
            "생태공원": CategoryType.NATURE,
            "지명": CategoryType.HISTORY,  # 지명의 경우 역사적 의미 부여
            "관람": CategoryType.ENTERTAINMENT,  # 관람활동
            "체험": CategoryType.ADVENTURE,  # 직접 체험하는 활동
            "봉우리": CategoryType.NATURE,
            "고지": CategoryType.NATURE,
            "절": CategoryType.HISTORY,
            "사찰": CategoryType.HISTORY,
            "도립공원": CategoryType.NATURE,
            "항구": CategoryType.HISTORY,  # 항구도 역사적, 문화적 가치로 볼 수 있음
            "관광농원": CategoryType.NATURE,
            "팜스테이": CategoryType.NATURE,
            "테마공원": CategoryType.ENTERTAINMENT,
            "강": CategoryType.NATURE,
            "하천": CategoryType.NATURE,
            "도시": CategoryType.CULTURE,  # 도시 관광은 문화 콘텐츠로 분류
            "해수욕장": CategoryType.NATURE,
            "해변": CategoryType.NATURE,
            "체험마을": CategoryType.CULTURE,  # 마을 단위 체험은 문화적 요소 포함
            "자연공원": CategoryType.NATURE,
            "도보코스": CategoryType.ADVENTURE,
            "온천": CategoryType.RELAXATION,
            "스파": CategoryType.RELAXATION,
            "산": CategoryType.NATURE,
            "동물원": CategoryType.ENTERTAINMENT,
            "유적지": CategoryType.HISTORY,
            "사적지": CategoryType.HISTORY,
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
