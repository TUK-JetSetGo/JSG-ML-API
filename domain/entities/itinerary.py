"""
Itinerary 엔티티
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

from domain.value_objects.time_value import TimeWindow


@dataclass
class DayRoute:
    """
    하루 일정의 '경로' (최적화 로직에서 활용)

    Attributes:
        day (int): 몇 일차인지 (정수)
        route (List[int]): 하루 동안 방문할 관광지의 ID 목록 (순서대로)
        daily_distance (float): 해당 일차에 이동한 총 거리 (km)
        daily_duration (float): 해당 일차에 소요된 총 시간 (시간 단위)
        start_time (Optional[time]): 일정 시작 시각
        end_time (Optional[time]): 일정 종료 시각
        visit_durations (Dict[int, float]): 각 관광지별 체류 시간
        travel_durations (Dict[str, float]): 이동 구간별 소요 시간 (예: "1->2": 0.5)
    """

    day: int
    route: List[int]
    daily_distance: float
    daily_duration: float
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    visit_durations: Dict[int, float] = field(default_factory=dict)
    travel_durations: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # day가 문자열인 경우 정수로 변환
        if isinstance(self.day, str):
            self.day = int(self.day)
        # daily_distance와 daily_duration이 문자열인 경우 실수로 변환
        if isinstance(self.daily_distance, str):
            self.daily_distance = float(self.daily_distance)
        if isinstance(self.daily_duration, str):
            self.daily_duration = float(self.daily_duration)
        # 모든 route 요소를 정수로 변환
        self.route = [int(spot) for spot in self.route]

    @property
    def time_window(self) -> Optional[TimeWindow]:
        """
        start_time과 end_time이 모두 존재할 경우, TimeWindow VO를 반환합니다.
        """
        if self.start_time is not None and self.end_time is not None:
            return TimeWindow(self.start_time, self.end_time)
        return None


@dataclass
class ItineraryItem:
    """
    하루 일정 내의 '방문' 또는 '이동' 항목 (사용자 일정표 요소)

    Attributes:
        item_number (int): 항목 순서 (1부터 시작)
        type (str): 항목 타입 ('visit' 또는 'travel')
        start_time (Optional[time]): 시작 시각
        end_time (Optional[time]): 종료 시각
        duration (float): 소요 시간 (시간 단위)
        spot_id (Optional[int]): 방문 관광지 ID (방문 시)
        spot_name (Optional[str]): 방문 관광지 이름
        from_spot_id (Optional[int]): 이동 출발지 ID (이동 시)
        from_spot_name (Optional[str]): 이동 출발지 이름
        to_spot_id (Optional[int]): 이동 도착지 ID (이동 시)
        to_spot_name (Optional[str]): 이동 도착지 이름
    """

    item_number: int
    type: str
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    duration: float = 0.0
    spot_id: Optional[int] = None
    spot_name: Optional[str] = None
    from_spot_id: Optional[int] = None
    from_spot_name: Optional[str] = None
    to_spot_id: Optional[int] = None
    to_spot_name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.duration, str):
            self.duration = float(self.duration)

    @property
    def time_window(self) -> Optional[TimeWindow]:
        """
        start_time과 end_time이 모두 존재할 경우, TimeWindow VO를 반환합니다.
        """
        if self.start_time is not None and self.end_time is not None:
            return TimeWindow(self.start_time, self.end_time)
        return None


@dataclass
class ItineraryDay:
    """
    하루치 여행 일정 (사용자에게 보여질 형태)

    Attributes:
        day_number (int): 전체 일정 중 몇 일차인지 (1부터 시작)
        items (List[ItineraryItem]): 해당 일차의 일정 항목들
    """

    day_number: int
    items: List[ItineraryItem] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.day_number, str):
            self.day_number = int(self.day_number)


@dataclass
class Itinerary:
    """
    전체 여행 일정 엔티티

    Attributes:
        daily_routes (List[DayRoute]): 최적화 로직을 위한 일자별 경로 정보
        days (List[ItineraryDay]): 사용자에게 보여줄 일자별 일정표
        overall_distance (float): 전체 이동 거리 (km)
        created_at (datetime): 생성 시각
        updated_at (datetime): 마지막 수정 시각
        start_date (Optional[date]): 여행 시작 날짜
        end_date (Optional[date]): 여행 종료 날짜
    """

    daily_routes: List[DayRoute] = field(default_factory=list)
    days: List[ItineraryDay] = field(default_factory=list)
    overall_distance: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def __post_init__(self):
        # dict 형태인 경우 객체로 변환
        self.daily_routes = [
            route if isinstance(route, DayRoute) else DayRoute(**route)
            for route in self.daily_routes
        ]
        self.days = [
            day if isinstance(day, ItineraryDay) else ItineraryDay(**day)
            for day in self.days
        ]
        if isinstance(self.overall_distance, str):
            self.overall_distance = float(self.overall_distance)

    @property
    def num_days(self) -> int:
        """총 여행 일수를 반환 (daily_routes와 days 중 큰 값)."""
        return max(len(self.daily_routes), len(self.days))

    def get_all_tourist_spots_from_routes(self) -> List[int]:
        """daily_routes에서 모든 관광지 ID를 중복 없이 반환합니다."""
        spots = [spot for route in self.daily_routes for spot in route.route]
        return list(set(spots))

    def get_day_route(self, day: int) -> Optional[DayRoute]:
        """특정 일차의 DayRoute를 반환합니다."""
        return next((route for route in self.daily_routes if route.day == day), None)

    def calculate_stats_from_routes(self) -> Dict[str, Any]:
        """daily_routes 기반 통계 계산."""
        total_spots = sum(len(route.route) for route in self.daily_routes)
        total_duration = sum(route.daily_duration for route in self.daily_routes)
        unique_spots = self.get_all_tourist_spots_from_routes()
        num_routes = len(self.daily_routes) or 1
        return {
            "total_days": len(self.daily_routes),
            "total_distance": self.overall_distance,
            "total_spots": total_spots,
            "unique_spots_count": len(unique_spots),
            "avg_spots_per_day": total_spots / num_routes,
            "avg_distance_per_day": self.overall_distance / num_routes,
            "avg_duration_per_day": total_duration / num_routes,
        }

    def calculate_stats_from_days(self) -> Dict[str, Any]:
        """days 기반 통계 계산."""
        total_items = sum(len(day.items) for day in self.days)
        return {
            "total_days": len(self.days),
            "total_items": total_items,
        }
