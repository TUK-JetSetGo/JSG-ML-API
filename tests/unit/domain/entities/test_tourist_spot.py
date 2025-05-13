from datetime import datetime, time

import pytest

from domain.entities.tourist_spot import TouristSpot
from domain.value_objects.coordinate import Coordinate


def test_post_init_type_conversion():
    ts = TouristSpot(
        tourist_spot_id="101",
        name="Test Spot",
        coordinate=("37.5665", "126.9780"),
        activity_level="0.75",
    )
    assert isinstance(ts.tourist_spot_id, int) and ts.tourist_spot_id == 101
    assert isinstance(ts.coordinate, Coordinate)
    assert ts.coordinate.latitude == pytest.approx(37.5665)
    assert ts.coordinate.longitude == pytest.approx(126.9780)
    assert isinstance(ts.activity_level, float) and ts.activity_level == pytest.approx(
        0.75
    )


def test_category_conversion_string_to_list():
    ts = TouristSpot(tourist_spot_id=1, name="Test", category="museum")
    assert isinstance(ts.category, list)
    assert ts.category == ["museum"]


def test_category_preserve_list():
    ts = TouristSpot(tourist_spot_id=1, name="Test", category=["a", "b"])
    assert ts.category == ["a", "b"]


def test_opening_hours_parsing_valid():
    ts = TouristSpot(tourist_spot_id=2, name="Test", opening_hours="07:00-20:00")
    expected_open = datetime.strptime("07:00", "%H:%M").time()
    expected_close = datetime.strptime("20:00", "%H:%M").time()
    assert ts.opening_time == expected_open
    assert ts.closing_time == expected_close


def test_opening_hours_parsing_invalid_format():
    ts = TouristSpot(tourist_spot_id=3, name="Test", opening_hours="invalid")
    assert ts.opening_time is None
    assert ts.closing_time is None


def test_opening_hours_none():
    ts = TouristSpot(tourist_spot_id=4, name="Test", opening_hours=None)
    assert ts.opening_time is None
    assert ts.closing_time is None


def test_is_open_at_true():
    ts = TouristSpot(tourist_spot_id=5, name="Test", opening_hours="08:00-18:00")
    assert ts.is_open_at(time(8, 0)) is True
    assert ts.is_open_at(time(12, 0)) is True
    assert ts.is_open_at(time(18, 0)) is True


def test_is_open_at_false():
    ts = TouristSpot(tourist_spot_id=6, name="Test", opening_hours="09:00-17:00")
    assert ts.is_open_at(time(8, 59)) is False
    assert ts.is_open_at(time(17, 1)) is False
    ts2 = TouristSpot(tourist_spot_id=7, name="Test", opening_hours=None)
    assert ts2.is_open_at(time(10, 0)) is False


def test_eq_and_str():
    ts1 = TouristSpot(tourist_spot_id=10, name="Name")
    ts2 = TouristSpot(tourist_spot_id=10, name="Different")
    ts3 = TouristSpot(tourist_spot_id=11, name="Name")
    assert ts1 == ts2
    assert ts1 != ts3
    assert str(ts1) == "TouristSpot(tourist_spot_id=10, name=Name)"
