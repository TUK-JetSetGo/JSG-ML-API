from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UserProfile:
    travel_type: str
    group_size: int
    budget_amount: int
    themes: List[str] = field(default_factory=list)
    must_visit_list: List[int] = field(default_factory=list)
    not_visit_list: List[int] = field(default_factory=list)
    preferred_transport: str = "car"

    id: int = 1  # TODO: 이거 나중에 DB 불러오도록 해야함.
    preferred_activity_level: Optional[float] = None  # 선호 활동 수준 (0-10)
    preferred_start_time: Optional[str] = None  # 선호 일일 시작 시간
    preferred_end_time: Optional[str] = None  # 선호 일일 종료 시간
    preferred_meal_times: Dict[str, str] = field(
        default_factory=dict
    )  # 선호 식사 시간 (예: {'breakfast': '08:00', 'lunch': '12:00', 'dinner': '18:00'})

    # 사용자 특성 벡터
    feature_vector: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """초기화 후 처리 - 데이터 타입 변환 등"""
        # 문자열로 들어온 group_size를 정수로 변환
        if isinstance(self.group_size, str):
            try:
                self.group_size = int(self.group_size)
            except (ValueError, TypeError):
                self.group_size = 1

        # 문자열로 들어온 budget_amount를 정수로 변환
        if isinstance(self.budget_amount, str):
            try:
                self.budget_amount = int(self.budget_amount)
            except (ValueError, TypeError):
                self.budget_amount = 0

        # must_visit_list와 not_visit_list의 요소가 문자열이면 정수로 변환
        self.must_visit_list = [
            int(id) if isinstance(id, str) else id for id in self.must_visit_list
        ]
        self.not_visit_list = [
            int(id) if isinstance(id, str) else id for id in self.not_visit_list
        ]

    def generate_feature_vector(self) -> Dict[str, float]:
        """사용자 프로필에서 특성 벡터 생성"""
        features = {}

        # 여행 유형 원-핫 인코딩
        travel_types = ["family", "solo", "couple", "friends", "business"]
        for t_type in travel_types:
            features[f"travel_type_{t_type}"] = (
                1.0 if self.travel_type == t_type else 0.0
            )

        # 그룹 크기 정규화 (1-10 범위로 가정)
        features["group_size"] = min(1.0, self.group_size / 10.0)

        # 예산 정규화 (최대 1,000,000으로 가정)
        features["budget"] = min(1.0, self.budget_amount / 1000000.0)

        # 테마 원-핫 인코딩
        common_themes = [
            "nature",
            "food",
            "culture",
            "shopping",
            "adventure",
            "relaxation",
            "history",
        ]
        for theme in common_themes:
            features[f"theme_{theme}"] = 1.0 if theme in self.themes else 0.0

        # 선호 교통 수단 원-핫 인코딩
        transport_types = ["car", "public_transport", "walk"]
        for transport in transport_types:
            features[f"transport_{transport}"] = (
                1.0 if self.preferred_transport == transport else 0.0
            )

        # 선호 활동 수준 정규화
        if self.preferred_activity_level is not None:
            features["activity_level"] = self.preferred_activity_level / 10.0
        else:
            features["activity_level"] = 0.5

        self.feature_vector = features
        return features
