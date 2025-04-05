"""
Itinerary 엔티티에 대한 테스트 코드
"""

from domain.entities.itinerary import (DayRoute, Itinerary, ItineraryDay,
                                       ItineraryItem)


def test_dayroute_post_init_conversion():
    """DayRoute: 문자열 입력이 올바르게 변환되는지 테스트"""
    day_route = DayRoute(
        day="1",
        route=["101", "102"],
        daily_distance="10.5",
        daily_duration="2.0",
        start_time=None,
        end_time=None,
    )
    assert isinstance(day_route.day, int) and day_route.day == 1
    assert (
        isinstance(day_route.daily_distance, float) and day_route.daily_distance == 10.5
    )
    assert (
        isinstance(day_route.daily_duration, float) and day_route.daily_duration == 2.0
    )
    # route 목록의 각 요소가 정수로 변환되어야 함
    assert day_route.route == [101, 102]


def test_dayroute_default_fields():
    """DayRoute: 기본값이 올바르게 설정되는지 테스트"""
    day_route = DayRoute(day=2, route=[1, 2], daily_distance=5.0, daily_duration=1.0)
    assert day_route.start_time is None
    assert day_route.end_time is None
    assert not day_route.visit_durations
    assert not day_route.travel_durations


def test_itinerary_item_duration_conversion():
    """ItineraryItem: duration이 문자열일 때 실수로 변환되는지 테스트"""
    item = ItineraryItem(item_number=1, type="visit", duration="1.5", spot_id=101)
    assert isinstance(item.duration, float)
    assert item.duration == 1.5


def test_itinerary_item_default_values():
    """ItineraryItem: 기본 값들이 올바르게 설정되는지 테스트"""
    item = ItineraryItem(item_number=2, type="travel")
    assert item.start_time is None
    assert item.end_time is None
    assert item.duration == 0.0
    assert item.spot_id is None
    assert item.from_spot_id is None


def test_itinerary_day_post_init_conversion():
    """ItineraryDay: day_number가 문자열일 때 정수로 변환되는지 테스트"""
    itinerary_day = ItineraryDay(day_number="3")
    assert isinstance(itinerary_day.day_number, int)
    assert itinerary_day.day_number == 3


def test_itinerary_day_default_items():
    """ItineraryDay: items가 기본값으로 빈 리스트인지 테스트"""
    itinerary_day = ItineraryDay(day_number=1)
    assert not itinerary_day.items


def test_itinerary_post_init_conversion_and_property_num_days():
    """Itinerary: __post_init__에서 dict 입력이 객체로 변환되고, num_days 프로퍼티가 계산되는지 테스트"""
    day_route_dict = {
        "day": "1",
        "route": ["101", "102"],
        "daily_distance": "10.0",
        "daily_duration": "2.0",
    }
    itinerary_day_dict = {"day_number": "1", "items": []}
    itinerary = Itinerary(
        daily_routes=[day_route_dict],
        days=[itinerary_day_dict],
        overall_distance="20.0",
    )
    assert isinstance(itinerary.daily_routes[0], DayRoute)
    assert itinerary.daily_routes[0].day == 1
    assert itinerary.overall_distance == 20.0
    # num_days: daily_routes와 days 중 큰 값 (여기서는 둘 다 길이 1)
    assert itinerary.num_days == 1


def test_get_all_tourist_spots_from_routes():
    """Itinerary: 모든 관광지 ID가 중복 없이 반환되는지 테스트"""
    day_route1 = DayRoute(
        day=1, route=[1, 2, 3], daily_distance=5.0, daily_duration=1.0
    )
    day_route2 = DayRoute(
        day=2, route=[3, 4, 5], daily_distance=6.0, daily_duration=1.5
    )
    itinerary = Itinerary(daily_routes=[day_route1, day_route2])
    spots = itinerary.get_all_tourist_spots_from_routes()
    # 반환된 결과는 순서에 상관없이 [1, 2, 3, 4, 5]여야 함
    assert sorted(spots) == [1, 2, 3, 4, 5]


def test_get_day_route_found_and_not_found():
    """Itinerary: 특정 일차의 DayRoute가 올바르게 반환되는지 테스트"""
    day_route1 = DayRoute(day=1, route=[10, 20], daily_distance=7.0, daily_duration=2.0)
    day_route2 = DayRoute(day=2, route=[30, 40], daily_distance=8.0, daily_duration=2.5)
    itinerary = Itinerary(daily_routes=[day_route1, day_route2])
    found_route = itinerary.get_day_route(2)
    not_found_route = itinerary.get_day_route(3)
    assert found_route == day_route2
    assert not_found_route is None


def test_calculate_stats_from_routes():
    """Itinerary: daily_routes 기반 통계 계산이 올바른지 테스트"""
    day_route1 = DayRoute(day=1, route=[1, 2], daily_distance=10.0, daily_duration=2.0)
    day_route2 = DayRoute(
        day=2, route=[2, 3, 4], daily_distance=15.0, daily_duration=3.0
    )
    itinerary = Itinerary(daily_routes=[day_route1, day_route2], overall_distance=25.0)
    stats = itinerary.calculate_stats_from_routes()
    assert stats["total_days"] == 2
    assert stats["total_distance"] == 25.0
    # total_spots = len(day_route1.route) + len(day_route2.route) = 2 + 3 = 5
    assert stats["total_spots"] == 5
    # unique_spots: [1, 2] U [2, 3, 4] = [1, 2, 3, 4]
    assert stats["unique_spots_count"] == 4
    assert stats["avg_spots_per_day"] == 5 / 2
    assert stats["avg_distance_per_day"] == 25.0 / 2
    # avg_duration_per_day = (2.0 + 3.0) / 2 = 2.5
    assert stats["avg_duration_per_day"] == 2.5


def test_calculate_stats_from_days():
    """Itinerary: days 기반 통계 계산이 올바른지 테스트"""
    item1 = ItineraryItem(item_number=1, type="visit", duration=1.0, spot_id=101)
    item2 = ItineraryItem(item_number=2, type="travel", duration=0.5)
    itinerary_day1 = ItineraryDay(day_number=1, items=[item1, item2])
    itinerary_day2 = ItineraryDay(day_number=2, items=[item1])
    itinerary = Itinerary(days=[itinerary_day1, itinerary_day2])
    stats = itinerary.calculate_stats_from_days()
    assert stats["total_days"] == 2
    # total_items: day1: 2, day2: 1 => 3
    assert stats["total_items"] == 3
