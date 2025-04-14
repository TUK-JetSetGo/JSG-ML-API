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
    route, total_dist, total_dur = RouteOptimizationService.optimize_single_day_route(
        spots=[],
        start_spot_index=0,
        base_scores={},
        priority_scores={},
        max_distance=10,
        max_duration=5,
        max_places=3,
    )
    assert route == []
    assert total_dist == 0.0
    assert total_dur == 0.0


def test_optimize_single_day_route_basic(simple_spots):
    """
    간단한 3개 관광지에서 단일 일자 경로 최적화가 올바르게 작동하는지 테스트합니다.
    - 시작 관광지는 인덱스 0으로 설정.
    - 충분한 이동 거리 및 소요 시간 제한을 부여.
    - 반환된 경로는 시작지점이 맨 앞과 맨 뒤에 위치해야 합니다.
    """
    base_scores = {1: 10.0, 2: 20.0, 3: 15.0}
    priority_scores = {}
    max_distance = 20.0
    max_duration = 5.0
    max_places = 3

    route, total_dist, total_dur = RouteOptimizationService.optimize_single_day_route(
        spots=simple_spots,
        start_spot_index=0,
        base_scores=base_scores,
        priority_scores=priority_scores,
        max_distance=max_distance,
        max_duration=max_duration,
        max_places=max_places,
    )

    # 반환된 경로의 시작점과 끝은 동일해야 함.
    assert route[0] == 0
    assert route[-1] == 0
    # 방문한 장소 수가 제한(max_places)을 초과하지 않아야 함.
    assert len(route) <= max_places + 1  # 시작/종료 포함
    # 총 이동거리와 소요 시간이 지정 한계 내에 있어야 함.
    assert total_dist <= max_distance + 1e-3
    assert total_dur <= max_duration + 1e-3


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

    route, total_dist, total_dur = RouteOptimizationService.optimize_single_day_route(
        spots=spots,
        start_spot_index=0,
        base_scores=base_scores,
        priority_scores=priority_scores,
        max_distance=max_distance,
        max_duration=max_duration,
        max_places=max_places,
        must_visit_indices=must_visit_indices,
    )

    # 반환된 경로(시작, 방문, 종료)에서 반드시 방문해야 하는 관광지가 포함되어 있는지 확인
    visited = route[1:-1]
    must_visit_id = spots[1].id
    visited_ids = [spots[idx].id for idx in visited]
    assert (
        must_visit_id in visited_ids
    ), f"Must-visit spot id {must_visit_id} not found in visited_ids: {visited_ids}"


def test_optimize_multi_day_route():
    """
    클러스터별 관광지와 사용자 프로필을 기반으로 다중 일자 경로 최적화가 올바르게 수행되는지 테스트합니다.
    클러스터 1과 클러스터 2로 구성하고, 각 클러스터에서 지정된 시작지점 및 must_visit 관광지를 확인합니다.
    """
    # 클러스터 정의 (production Coordinate.distance_to를 고려하면, (0,1) 간 거리는 약 111.32km)
    cluster1 = [
        DummyTouristSpot(id=1, latitude=0, longitude=0, average_visit_duration=0.5),
        DummyTouristSpot(id=2, latitude=0, longitude=1, average_visit_duration=0.5),
    ]
    cluster2 = [
        DummyTouristSpot(id=3, latitude=0, longitude=2, average_visit_duration=0.5),
        DummyTouristSpot(id=4, latitude=0, longitude=3, average_visit_duration=0.5),
    ]
    clusters = {1: cluster1, 2: cluster2}

    user_profile = DummyUserProfile(must_visit_list=[2, 3])

    num_days = 2
    max_places_per_day = 2
    # daily_start_points에는 각 클러스터에서 시작할 관광지 id를 지정
    daily_start_points = [1, 3]

    # production 코드에서는 위도/경도 기준 거리 계산 시 (0,0)-(0,1) 거리는 약 111.32km 정도이므로
    # 충분한 daily_max_distance와 daily_max_duration 값을 지정해야 최적화가 정상 동작합니다.
    daily_max_distance = 1000.0  # km
    daily_max_duration = 50.0  # 시간
    base_scores = {1: 10.0, 2: 20.0, 3: 15.0, 4: 10.0}
    priority_scores = {}

    daily_routes = RouteOptimizationService.optimize_multi_day_route(
        clusters=clusters,
        user_profile=user_profile,
        num_days=num_days,
        max_places_per_day=max_places_per_day,
        daily_start_points=daily_start_points,
        daily_max_distance=daily_max_distance,
        daily_max_duration=daily_max_duration,
        base_scores=base_scores,
        priority_scores=priority_scores,
    )
    assert len(daily_routes) == num_days

    # 각 일자에 대해 생성된 경로가 존재하고, 시작/종료가 동일하며, 이동 거리 및 소요 시간이 제한 내에 있는지 검증
    for day_route, total_dist, total_dur in daily_routes:
        assert len(day_route) > 0, "생성된 경로가 비어 있습니다."
        assert (
            day_route[0] == day_route[-1]
        ), f"경로의 시작 {day_route[0]}과 종료 {day_route[-1]}가 다릅니다."
        assert (
            total_dist <= daily_max_distance + 1e-3
        ), f"총 이동 거리가 제한을 초과합니다: {total_dist} km"
        assert (
            total_dur <= daily_max_duration + 1e-3
        ), f"총 소요 시간이 제한을 초과합니다: {total_dur} 시간"
