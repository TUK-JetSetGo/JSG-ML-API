# pylint: disable=import-error
"""
vo time_value에 대한 유닛테스트 코드
"""
from datetime import time

import pytest

from domain.value_objects.time_value import TimeWindow, TravelDuration


def test_time_window_duration_normal():
    """정상 시간 범위 (같은 날)에서의 지속 시간이 올바른지 확인."""
    start = time(9, 0)
    end = time(17, 0)
    time_window = TimeWindow(start, end)
    assert time_window.duration_hours() == 8


def test_time_window_duration_midnight():
    """종료 시간이 다음 날인 경우 (예: 22:00 ~ 02:00)의 지속 시간을 확인."""
    start = time(22, 0)
    end = time(2, 0)
    time_window = TimeWindow(start, end)

    # 22:00~24:00: 2시간, 00:00~02:00: 2시간 => 총 4시간
    assert pytest.approx(time_window.duration_hours(), rel=1e-3) == 4


def test_time_window_contains_normal():
    """정상 시간 범위 내에 특정 시간이 포함되는지 검증."""
    time_window = TimeWindow(time(9, 0), time(17, 0))
    # 시작과 종료 포함, 중간 시간 포함
    assert time_window.contains(time(9, 0))
    assert time_window.contains(time(12, 0))
    assert time_window.contains(time(17, 0))
    # 범위를 벗어난 시간은 False
    assert not time_window.contains(time(8, 59))
    assert not time_window.contains(time(17, 1))


def test_time_window_contains_midnight():
    """종료 시간이 다음 날인 경우, 포함 여부를 확인."""
    time_window = TimeWindow(time(22, 0), time(2, 0))
    # 22:00 ~ 02:00 범위 내의 시간들
    assert time_window.contains(time(22, 0))
    assert time_window.contains(time(23, 30))
    assert time_window.contains(time(0, 30))
    assert time_window.contains(time(1, 59))
    # 범위를 벗어난 시간들
    assert not time_window.contains(time(2, 1))
    assert not time_window.contains(time(21, 59))


def test_time_window_overlaps():
    """두 시간 범위가 겹치는지 여부를 검증."""
    # 동일 날, 겹치는 경우
    time_window1 = TimeWindow(time(9, 0), time(17, 0))
    time_window2 = TimeWindow(time(16, 0), time(18, 0))
    assert time_window1.overlaps(time_window2)
    assert time_window2.overlaps(time_window1)

    # 겹치지 않는 경우
    time_window3 = TimeWindow(time(9, 0), time(11, 0))
    time_window4 = TimeWindow(time(11, 1), time(13, 0))
    assert not time_window3.overlaps(time_window4)
    assert not time_window4.overlaps(time_window3)

    # 한쪽은 자정 넘어가는 경우
    time_window5 = TimeWindow(time(22, 0), time(2, 0))
    time_window6 = TimeWindow(time(1, 0), time(3, 0))
    assert time_window5.overlaps(time_window6)
    assert time_window6.overlaps(time_window5)


def test_time_window_str():
    """__str__ 메서드가 올바른 문자열을 반환하는지 검증."""
    time_window = TimeWindow(time(9, 0), time(17, 0))
    assert str(time_window) == "09:00 - 17:00"


def test_travel_duration_minutes_speed_and_str():
    """TravelDuration의 minutes, speed_kmh, __str__ 기능 검증."""
    travel_duration = TravelDuration(hours=1.5, distance_km=30, transport_mode="car")
    # 1.5시간은 90분, 평균 속도는 30 / 1.5 = 20 km/h
    assert travel_duration.minutes == 90
    assert pytest.approx(travel_duration.speed_kmh, rel=1e-3) == 20
    expected_str = "1시간 30분 (30.0km, car)"
    assert str(travel_duration) == expected_str


def test_travel_duration_str_less_than_one_hour():
    """1시간 미만의 TravelDuration 문자열 표현 검증."""
    travel_duration = TravelDuration(hours=0.5, distance_km=5, transport_mode="walk")
    # 0.5시간은 30분
    expected_str = "30분 (5.0km, walk)"
    assert str(travel_duration) == expected_str


def test_travel_duration_zero_hours():
    """시간이 0인 경우 속도는 0이어야 함."""
    travel_duration = TravelDuration(hours=0.0, distance_km=10, transport_mode="car")
    assert travel_duration.speed_kmh == 0
    # 문자열 표현은 0분으로 표기됨
    expected_str = "0분 (10.0km, car)"
    assert str(travel_duration) == expected_str
