"""
RouteOptimizaitonService에 대한 유닛테스트
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, invalid-name

import math

import pytest

from domain.services.route_optimization_service import RouteOptimizationService


class DummyCoordinate:
    """
    단순 유클리드 거리 계산을 위한 Dummy 좌표 클래스.
    실제 서비스에서는 보다 정교한 거리 계산(Haversine 등)을 사용할 수 있음.
    """

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
        return math.sqrt(
            (self.latitude - other.latitude) ** 2
            + (self.longitude - other.longitude) ** 2
        )


class DummyTouristSpot:
    """
    테스트용 관광지 클래스.
    - id: 관광지 고유 ID
    - coordinate: 좌표
    - average_visit_duration: 관광지 방문에 소요되는 시간 (시간 단위)
    """

    def __init__(self, id, latitude, longitude, average_visit_duration=1.0):
        self.id = id
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.average_visit_duration = average_visit_duration


class DummyUserProfile:
    """
    테스트용 사용자 프로필 클래스.
    - must_visit_list: 반드시 방문해야 하는 관광지의 id 리스트
    """

    def __init__(self, must_visit_list=None):
        if must_visit_list is None:
            must_visit_list = []
        self.must_visit_list = must_visit_list


@pytest.fixture
def simple_spots():
    """
    3개 관광지 리스트: 좌표는 (0,0), (0,3), (0,6)으로 설정.
    각 관광지 방문 시간은 0.5시간으로 설정.
    """
    spots = [
        DummyTouristSpot(id=1, latitude=0, longitude=0, average_visit_duration=0.5),
        DummyTouristSpot(id=2, latitude=0, longitude=3, average_visit_duration=0.5),
        DummyTouristSpot(id=3, latitude=0, longitude=6, average_visit_duration=0.5),
    ]
    return spots


def test_calculate_distance_matrix(simple_spots):
    """
    관광지 리스트에 대해 거리 행렬이 올바르게 계산되는지 테스트합니다.

    production의 Coordinate.distance_to 함수는 위도/경도를 기준으로
    실제 km 단위 거리를 반환합니다. (예: 적도에서 1도 ≒ 111.32km)

    따라서, (0,0) -> (0,3): 3 * 111.32 ≒ 333.96km,
              (0,0) -> (0,6): 6 * 111.32 ≒ 667.92km,
              (0,3) -> (0,6): 3 * 111.32 ≒ 333.96km 로 검증합니다.
    """
    matrix = RouteOptimizationService.calculate_distance_matrix(simple_spots)
    # 적도에서 1도의 경도당 거리 (km)
    km_per_degree = 111.31949079327357

    expected_0_1 = 3 * km_per_degree
    expected_0_2 = 6 * km_per_degree
    expected_1_2 = 3 * km_per_degree

    assert math.isclose(matrix[0][1], expected_0_1, rel_tol=1e-5)
    assert math.isclose(matrix[0][2], expected_0_2, rel_tol=1e-5)
    assert math.isclose(matrix[1][0], expected_0_1, rel_tol=1e-5)
    assert math.isclose(matrix[1][2], expected_1_2, rel_tol=1e-5)
    assert math.isclose(matrix[2][0], expected_0_2, rel_tol=1e-5)
    assert math.isclose(matrix[2][1], expected_1_2, rel_tol=1e-5)


def test_optimize_single_day_route_no_spots():
    """
    관광지 리스트가 빈 경우, 빈 경로와 0인 거리 및 소요 시간이 반환되는지 확인합니다.
    """
    service = RouteOptimizationService()
    route, total_dist, total_dur = service.optimize_route(
        spots=[],
        start_spot_index=0,
        base_scores={},
        priority_scores={},
        max_distance=10,
        max_duration=5,
        max_places=3,
        must_visit_indices=[],
        transport_speed_kmh=40.0,
    )
    assert route == []
    assert total_dist == 0.0
    assert total_dur == 0.0


def test_optimize_single_day_route_must_visit():
    """
    반드시 방문해야 하는 관광지가 경로에 포함되는지 테스트합니다.
    이 테스트에서는 원본 spots 리스트에서 두 번째 관광지(id=2)를 must_visit으로 지정합니다.
    """
    spots = [
        DummyTouristSpot(id=1, latitude=0, longitude=0, average_visit_duration=0.5),
        DummyTouristSpot(id=2, latitude=0, longitude=2, average_visit_duration=0.5),
        DummyTouristSpot(id=3, latitude=0, longitude=4, average_visit_duration=0.5),
    ]
    base_scores = {1: 10.0, 2: 5.0, 3: 5.0}
    priority_scores = {}
    max_distance = 1000.0
    max_duration = 50.0
    max_places = 3

    must_visit_indices = [1]

    service = RouteOptimizationService()
    route, total_dist, total_dur = service.optimize_route(
        spots=spots,
        start_spot_index=0,
        base_scores=base_scores,
        priority_scores=priority_scores,
        max_distance=max_distance,
        max_duration=max_duration,
        max_places=max_places,
        must_visit_indices=must_visit_indices,
        transport_speed_kmh=40.0,
    )

    # 반환된 경로(시작, 방문, 종료)에서 반드시 방문해야 하는 관광지가 포함되어 있는지 확인
    visited = route[1:-1]
    must_visit_id = spots[1].id
    visited_ids = [spots[idx].id for idx in visited]
    assert (
        must_visit_id in visited_ids
    ), f"Must-visit spot id {must_visit_id} not found in visited_ids: {visited_ids}"
