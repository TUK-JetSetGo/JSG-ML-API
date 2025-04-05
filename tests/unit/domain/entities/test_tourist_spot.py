"""
도메인 TouristSpot에 대한 유닛테스트
"""

from datetime import datetime, time

from domain.entities.tourist_spot import TouristSpot


def test_post_init_type_conversion():
    """__post_init__: 문자열로 전달된 id, latitude, longitude, activity_level를 올바르게 변환하는지 테스트"""
    tourist_spot = TouristSpot(
        id="101",
        name="Test Spot",
        latitude="37.5665",
        longitude="126.9780",
        activity_level="0.75",
    )
    assert isinstance(tourist_spot.id, int) and tourist_spot.id == 101
    assert isinstance(tourist_spot.latitude, float) and tourist_spot.latitude == 37.5665
    assert (
        isinstance(tourist_spot.longitude, float) and tourist_spot.longitude == 126.9780
    )
    assert (
        isinstance(tourist_spot.activity_level, float)
        and tourist_spot.activity_level == 0.75
    )


def test_categories_conversion():
    """__post_init__: categories가 문자열이면 리스트로 변환하는지 테스트"""
    tourist_spot = TouristSpot(id=102, name="Category Test", categories="museum")
    assert isinstance(tourist_spot.categories, list)
    assert tourist_spot.categories == ["museum"]


def test_opening_hours_parsing_valid():
    """__post_init__: 유효한 opening_hours 문자열을 파싱하여 opening_time과 closing_time을 설정하는지 테스트"""
    tourist_spot = TouristSpot(
        id=103, name="Open Hours Test", opening_hours="07:00-20:00"
    )
    expected_open = datetime.strptime("07:00", "%H:%M").time()
    expected_close = datetime.strptime("20:00", "%H:%M").time()
    assert tourist_spot.opening_time == expected_open
    assert tourist_spot.closing_time == expected_close


def test_opening_hours_parsing_invalid():
    """__post_init__: 잘못된 opening_hours 문자열이 들어오면 opening_time과 closing_time이 None이 되는지 테스트"""
    tourist_spot = TouristSpot(
        id=104, name="Invalid Hours Test", opening_hours="invalid_format"
    )
    assert tourist_spot.opening_time is None
    assert tourist_spot.closing_time is None


def test_is_open_at_true():
    """is_open_at: 지정한 시간에 관광지가 개장 중이면 True를 반환하는지 테스트"""
    tourist_spot = TouristSpot(id=105, name="Open Test", opening_hours="08:00-18:00")
    check_time = datetime.strptime("12:00", "%H:%M").time()
    assert tourist_spot.is_open_at(check_time) is True


def test_is_open_at_false():
    """is_open_at: 지정한 시간이 개장 시간이 아니면 False를 반환하는지 테스트"""
    tourist_spot = TouristSpot(id=106, name="Closed Test", opening_hours="09:00-17:00")
    check_time = datetime.strptime("08:00", "%H:%M").time()
    assert tourist_spot.is_open_at(check_time) is False


def test_is_open_at_no_hours():
    """is_open_at: opening_hours가 설정되지 않은 경우 False를 반환하는지 테스트"""
    tourist_spot = TouristSpot(id=107, name="No Hours Test", opening_hours=None)
    check_time = time(10, 0)
    assert tourist_spot.is_open_at(check_time) is False


def test_eq_method():
    """__eq__: 두 Touristourist_spotpot 객체가 id로 비교되어 동일성을 판단하는지 테스트"""
    tourist_spot1 = TouristSpot(id=108, name="Spot A")
    tourist_spot2 = TouristSpot(id=108, name="Spot A Different Name")
    tourist_spot3 = TouristSpot(id=109, name="Spot B")
    assert tourist_spot1 == tourist_spot2
    assert tourist_spot1 != tourist_spot3


def test_str_method():
    """__str__: __str__ 메서드가 id와 name을 포함한 문자열을 반환하는지 테스트"""
    tourist_spot = TouristSpot(id=110, name="String Test")
    expected_str = "TouristSpot(id=110, name=String Test)"
    assert str(tourist_spot) == expected_str
