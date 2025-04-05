"""
TouristSpot 엔티티
"""

from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import List, Optional

# VO 적용: Coordinate를 import 합니다.
from domain.value_objects.coordinate import Coordinate


@dataclass
class TouristSpot:
    id: int
    name: str
    # 기존의 위도/경도 대신 하나의 Coordinate VO 사용 (기본값은 (0.0, 0.0))
    coordinate: Coordinate = field(default_factory=lambda: Coordinate(0.0, 0.0))
    activity_level: float = 0.0
    address: Optional[str] = None
    business_hours: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    home_page: Optional[str] = None
    booking_url: Optional[str] = None
    tel: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_urls: List[str] = field(default_factory=list)
    travel_city_id: Optional[int] = None

    # 관광지 방문에 필요한 평균 체류 시간 (시간 단위)
    average_visit_duration: float = 1.0

    # 개장/폐장 시간 (파싱 결과)
    opening_time: Optional[time] = None
    closing_time: Optional[time] = None

    # 개장 시간 문자열 (예: "07:00-20:00")
    opening_hours: Optional[str] = None

    def __post_init__(self):
        # ID, activity_level, categories의 타입 변환
        if isinstance(self.id, str):
            self.id = int(self.id)
        if isinstance(self.activity_level, str):
            try:
                self.activity_level = float(self.activity_level)
            except (ValueError, TypeError):
                self.activity_level = 0.0
        if isinstance(self.categories, str):
            self.categories = [self.categories]

        # opening_hours 파싱
        if self.opening_hours:
            try:
                times = self.opening_hours.split("-")
                if len(times) == 2:
                    self.opening_time = datetime.strptime(
                        times[0].strip(), "%H:%M"
                    ).time()
                    self.closing_time = datetime.strptime(
                        times[1].strip(), "%H:%M"
                    ).time()
                else:
                    self.opening_time = None
                    self.closing_time = None
            except Exception:
                self.opening_time = None
                self.closing_time = None

        # coordinate 변환 처리:
        # coordinate 필드가 Coordinate 인스턴스가 아니라면, (lat, lon) 튜플이나 리스트 등으로 간주하여 변환합니다.
        if not isinstance(self.coordinate, Coordinate):
            try:
                lat, lon = self.coordinate  # 예: (latitude, longitude)
                self.__dict__["coordinate"] = Coordinate(float(lat), float(lon))
            except Exception:
                self.__dict__["coordinate"] = Coordinate(0.0, 0.0)

    def is_open_at(self, check_time: time) -> bool:
        """
        지정된 시간에 관광지가 개장 중인지 확인합니다.
        개장 시간이나 폐장 시간이 None이면 False를 반환합니다.
        """
        if self.opening_time is None or self.closing_time is None:
            return False
        return self.opening_time <= check_time <= self.closing_time

    def __eq__(self, other):
        """
        두 관광지 객체는 id가 같으면 동일한 객체로 간주합니다.
        """
        if not isinstance(other, TouristSpot):
            return False
        return self.id == other.id

    def __str__(self):
        """
        관광지의 문자열 표현은 id와 name을 포함합니다.
        """
        return f"TouristSpot(id={self.id}, name={self.name})"
