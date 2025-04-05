"""
Coordinate vo 클래스 테스트
"""

import math

from domain.value_objects.coordinate import Coordinate


def test_distance_to_same_point():
    """동일한 좌표 간의 거리는 0이어야 합니다."""
    coord1 = Coordinate(latitude=40.7128, longitude=-74.0060)
    coord2 = Coordinate(latitude=40.7128, longitude=-74.0060)
    distance = coord1.distance_to(coord2)
    assert math.isclose(distance, 0.0, abs_tol=1e-6)


def test_distance_to_known_cities():
    """New York와 London 사이의 거리를 대략적으로 검증합니다."""
    # New York: 40.7128° N, -74.0060° W
    # London: 51.5074° N, -0.1278° W
    new_york = Coordinate(latitude=40.7128, longitude=-74.0060)
    london = Coordinate(latitude=51.5074, longitude=-0.1278)
    distance = new_york.distance_to(london)
    # 실제 거리는 약 5570 km 정도입니다. (오차 범위 약 ±50km)
    assert 5500 <= distance <= 5650


def test_is_near_true():
    """is_near: 임계값 1km 이내이면 True를 반환하는지 테스트합니다."""
    # 서울 좌표
    seoul = Coordinate(latitude=37.5665, longitude=126.9780)
    # 오차
    nearby = Coordinate(latitude=37.5667, longitude=126.9782)
    assert seoul.is_near(nearby) is True


def test_is_near_false():
    """is_near: 임계값 1km보다 멀면 False를 반환하는지 테스트합니다."""
    # 서울과 부산의 좌표 예시
    seoul = Coordinate(latitude=37.5665, longitude=126.9780)
    busan = Coordinate(latitude=35.1796, longitude=129.0756)
    # 서울과 부산 사이 거리는 약 325km 정도
    assert seoul.is_near(busan) is False


def test_str_method():
    """__str__: 좌표의 문자열 표현이 올바른지 테스트합니다."""
    coord = Coordinate(latitude=37.5665, longitude=126.9780)
    expected_str = "(37.566500, 126.978000)"
    assert str(coord) == expected_str
