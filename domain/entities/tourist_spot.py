from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import List, Optional

from domain.value_objects.coordinate import Coordinate


@dataclass
class TouristSpot:
    tourist_spot_id: int
    name: str
    coordinate: Coordinate = field(default_factory=lambda: Coordinate(0.0, 0.0))
    activity_level: float = 0.0
    address: Optional[str] = None
    business_status: Optional[str] = None
    business_hours: Optional[str] = None
    category: List[str] = field(default_factory=list)
    home_page: Optional[str] = None
    naver_booking_url: Optional[str] = None
    tel: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_urls: List[str] = field(default_factory=list)
    travel_city_id: Optional[int] = None
    average_visit_duration: float = 1.0
    opening_time: Optional[time] = None
    closing_time: Optional[time] = None
    opening_hours: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.tourist_spot_id, str):
            try:
                self.tourist_spot_id = int(self.tourist_spot_id)
            except ValueError:
                self.tourist_spot_id = 0

        if isinstance(self.activity_level, str):
            try:
                self.activity_level = float(self.activity_level)
            except (ValueError, TypeError):
                self.activity_level = 0.0

        if isinstance(self.category, str):
            self.category = [self.category]

        if self.opening_hours:
            try:
                times = self.opening_hours.split("-")
                if len(times) == 2:
                    start_str, end_str = times[0].strip(), times[1].strip()
                    self.opening_time = datetime.strptime(start_str, "%H:%M").time()
                    self.closing_time = datetime.strptime(end_str, "%H:%M").time()
                else:
                    self.opening_time = None
                    self.closing_time = None
            except Exception:
                self.opening_time = None
                self.closing_time = None

        if not isinstance(self.coordinate, Coordinate):
            try:
                lat, lon = self.coordinate
                self.coordinate = Coordinate(float(lat), float(lon))
            except Exception:
                self.coordinate = Coordinate(0.0, 0.0)

    def is_open_at(self, check_time: time) -> bool:
        if self.opening_time is None or self.closing_time is None:
            return False
        return self.opening_time <= check_time <= self.closing_time

    def __eq__(self, other):
        if not isinstance(other, TouristSpot):
            return False
        return self.tourist_spot_id == other.tourist_spot_id

    def __str__(self):
        return f"TouristSpot(tourist_spot_id={self.tourist_spot_id}, name={self.name})"
