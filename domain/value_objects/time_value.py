"""
시간 관련 값 객체 모듈
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta


@dataclass(frozen=True)
class TimeWindow:
    """시간 범위를 나타내는 값 객체"""

    start_time: time
    end_time: time

    def __post_init__(self):
        """시작 시간이 종료 시간보다 늦은 경우 검증"""
        # 시작 시간과 종료 시간을 비교하기 위해 같은 날짜의 datetime 객체로 변환
        start_dt = datetime.combine(datetime.today(), self.start_time)
        end_dt = datetime.combine(datetime.today(), self.end_time)

        # 종료 시간이 다음 날인 경우(예: 22:00 ~ 02:00)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)

    def duration_hours(self) -> float:
        """시간 범위의 지속 시간을 시간 단위로 반환"""
        start_dt = datetime.combine(datetime.today(), self.start_time)
        end_dt = datetime.combine(datetime.today(), self.end_time)

        # 종료 시간이 다음 날인 경우(예: 22:00 ~ 02:00)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)

        duration = end_dt - start_dt
        return duration.total_seconds() / 3600  # 초를 시간으로 변환

    def contains(self, check_time: time) -> bool:
        """주어진 시간이 시간 범위 내에 있는지 확인"""
        t_dt = datetime.combine(datetime.today(), check_time)
        start_dt = datetime.combine(datetime.today(), self.start_time)
        end_dt = datetime.combine(datetime.today(), self.end_time)

        # 종료 시간이 다음 날인 경우(예: 22:00 ~ 02:00)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
            # 시작 시간보다 이전이면 다음 날로 취급
            if t_dt < start_dt:
                t_dt += timedelta(days=1)

        return start_dt <= t_dt <= end_dt

    def overlaps(self, other: "TimeWindow") -> bool:
        """다른 시간 범위와 겹치는지 확인"""
        return (
            self.contains(other.start_time)
            or self.contains(other.end_time)
            or other.contains(self.start_time)
            or other.contains(self.end_time)
        )

    def __str__(self) -> str:
        return (
            f"{self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')}"
        )


@dataclass(frozen=True)
class TravelDuration:
    """이동 시간을 나타내는 값 객체"""

    hours: float  # 시간 단위
    distance_km: float  # 거리 (km)
    transport_mode: str  # 교통 수단 (car, public_transport, walk)

    @property
    def minutes(self) -> int:
        """시간을 분 단위로 반환"""
        return int(self.hours * 60)

    @property
    def speed_kmh(self) -> float:
        """평균 속도 (km/h)"""
        if self.hours > 0:
            return self.distance_km / self.hours
        return 0

    def __str__(self) -> str:
        hours_int = int(self.hours)
        minutes_int = int((self.hours - hours_int) * 60)
        if hours_int > 0:
            return (
                f"{hours_int}시간 {minutes_int}분 "
                f"({self.distance_km:.1f}km, {self.transport_mode})"
            )
        return f"{minutes_int}분 ({self.distance_km:.1f}km, {self.transport_mode})"
