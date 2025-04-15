import math
import unittest
from typing import Dict, List

import pulp

from infrastructure.adapters.ml.enhanced_ptppp_model import EnhancedPTPPPModel


# Dummy 클래스 정의
class DummyCoordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
        # 테스트용으로 간단한 유클리드 거리 계산 (1도 당 약 111km로 환산)
        lat_diff = self.latitude - other.latitude
        lon_diff = self.longitude - other.longitude
        return math.sqrt(lat_diff**2 + lon_diff**2) * 111


class DummyTouristSpot:
    def __init__(
        self,
        tourist_spot_id: int,
        latitude: float,
        longitude: float,
        average_visit_duration: float = 1.0,
    ):
        self.tourist_spot_id = tourist_spot_id
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.average_visit_duration = average_visit_duration
        # 내부 모델에서는 spot.id에 접근하므로 동일하게 할당
        self.id = tourist_spot_id


class DummyUserProfile:
    def __init__(self, must_visit_list: List[int] = None):
        # 반드시 방문해야 하는 관광지 ID 리스트 (예: [2])
        self.must_visit_list = must_visit_list or []


class TestEnhancedPTPPPModel(unittest.TestCase):

    def setUp(self):
        # 4개의 DummyTouristSpot을 생성 (좌표가 다르게 설정됨)
        self.spots = [
            DummyTouristSpot(1, 37.0, 127.0, average_visit_duration=0.5),
            DummyTouristSpot(2, 37.01, 127.01, average_visit_duration=0.75),
            DummyTouristSpot(3, 37.02, 127.02, average_visit_duration=0.6),
            DummyTouristSpot(4, 37.03, 127.03, average_visit_duration=0.8),
        ]
        # 사용자 프로필: 반드시 방문해야 하는 관광지는 id 2
        self.user_profile = DummyUserProfile(must_visit_list=[2])
        # Base scores (관광지 id를 키로 하는 기본 점수)
        self.base_scores: Dict[int, float] = {1: 5.0, 2: 7.0, 3: 6.0, 4: 8.0}
        # Priority scores (순서별 추가 점수: 첫 번째 키는 방문 순서, 두 번째 키는 관광지 id)
        self.priority_scores: Dict[int, Dict[int, float]] = {
            1: {1: 1.0, 2: 2.0, 3: 1.5, 4: 2.5},
            2: {1: 0.5, 2: 1.0, 3: 0.8, 4: 1.2},
            3: {1: 0.3, 2: 0.7, 3: 0.5, 4: 0.9},
        }
        # 테스트 대상 모델 인스턴스 생성
        self.model = EnhancedPTPPPModel()

    def test_calculate_distance_matrix(self):
        # 관광지 간 거리 행렬 계산 테스트
        matrix = self.model.calculate_distance_matrix(self.spots)
        n = len(self.spots)
        self.assertEqual(len(matrix), n)
        for row in matrix:
            self.assertEqual(len(row), n)
        # 대각선은 0이어야 함
        for i in range(n):
            self.assertAlmostEqual(matrix[i][i], 0.0)
        # 0이 아닌 다른 값들은 양수여야 함
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertGreater(matrix[i][j], 0.0)

    def test_optimize_route(self):
        # 단일 일자 경로 최적화 테스트 (optimize_route)
        start_spot_index = 0  # 시작은 첫 번째 관광지
        max_distance = 50.0  # km 단위 (임의 값)
        max_duration = 5.0  # 시간 단위 (임의 값)
        max_places = 4  # 방문할 수 있는 관광지 수 (시작 지점 제외)
        # must_visit_indices는 입력 spot 리스트의 인덱스 기준 – 여기서는 spot with id=2 (spots[1])를 강제 방문
        must_visit_indices = [1]
        transport_speed_kmh = 40.0

        route, total_distance, total_duration = self.model.optimize_route(
            spots=self.spots,
            start_spot_index=start_spot_index,
            base_scores=self.base_scores,
            priority_scores=self.priority_scores,
            max_distance=max_distance,
            max_duration=max_duration,
            max_places=max_places,
            must_visit_indices=must_visit_indices,
            transport_speed_kmh=transport_speed_kmh,
        )
        # route는 시작 지점 인덱스로 시작하고 종료되어야 함
        self.assertIsInstance(route, list)
        self.assertGreaterEqual(len(route), 2)
        self.assertEqual(route[0], start_spot_index)
        self.assertEqual(route[-1], start_spot_index)
        # 총 이동 거리와 소요 시간은 0 이상이어야 함
        self.assertGreaterEqual(total_distance, 0.0)
        self.assertGreaterEqual(total_duration, 0.0)
        # 반드시 방문해야 하는 관광지 (id 2, spots[1])가 경로에 포함되었는지 확인
        self.assertIn(1, route)

    def test_optimize_multi_day_route(self):
        # 다중 일자 경로 최적화 테스트 (optimize_multi_day_route)
        # 각 일자별로 동일한 관광지 목록 사용 (예제이므로 단순화)
        daily_spots: List[List[DummyTouristSpot]] = [self.spots, self.spots]
        # 각 일자의 시작 인덱스 (0번부터 시작)
        daily_start_indices = [0, 0]
        daily_max_distance = 50.0
        daily_max_duration = 5.0
        max_places_per_day = 4
        transport_speed_kmh = 40.0
        continuity_weight = 0.5

        multi_day_routes = self.model.optimize_multi_day_route(
            daily_spots=daily_spots,
            user_profile=self.user_profile,
            daily_start_indices=daily_start_indices,
            daily_max_distance=daily_max_distance,
            daily_max_duration=daily_max_duration,
            max_places_per_day=max_places_per_day,
            base_scores=self.base_scores,
            priority_scores=self.priority_scores,
            transport_speed_kmh=transport_speed_kmh,
            continuity_weight=continuity_weight,
        )
        # 반환된 결과는 일자 수 만큼 튜플 목록이어야 함
        self.assertIsInstance(multi_day_routes, list)
        self.assertEqual(len(multi_day_routes), len(daily_spots))
        for daily_route in multi_day_routes:
            # 각 daily_route는 (route, total_distance, total_duration)
            route_indices, day_distance, day_duration = daily_route
            self.assertIsInstance(route_indices, list)
            self.assertGreaterEqual(len(route_indices), 2)
            # 시작 및 종료 지점은 지정된 시작 인덱스(여기서는 0)이어야 함
            self.assertEqual(route_indices[0], 0)
            self.assertEqual(route_indices[-1], 0)
            self.assertGreaterEqual(day_distance, 0.0)
            self.assertGreaterEqual(day_duration, 0.0)

    def test_get_must_visit_indices(self):
        # 내부 메서드 _get_must_visit_indices 테스트
        must_visit = self.model._get_must_visit_indices(self.spots, self.user_profile)
        # user_profile.must_visit_list는 [2]이고, spots에서 id=2인 관광지는 인덱스 1에 위치해야 함
        self.assertIn(1, must_visit)

    def test_optimize_single_day_route_without_must_visit(self):
        # _optimize_single_day_route를 직접 호출하여 반드시 방문 제약이 없는 경우 테스트
        start_spot_index = 0
        max_distance = 50.0
        max_duration = 5.0
        max_places = 4
        transport_speed_kmh = 40.0

        route, total_distance, total_duration = self.model._optimize_single_day_route(
            spots=self.spots,
            start_spot_index=start_spot_index,
            base_scores=self.base_scores,
            priority_scores=self.priority_scores,
            max_distance=max_distance,
            max_duration=max_duration,
            max_places=max_places,
            must_visit_indices=[],  # 반드시 방문 제약 없음
            transport_speed_kmh=transport_speed_kmh,
            prev_day_end=None,
            continuity_weight=0.5,
        )
        self.assertIsInstance(route, list)
        self.assertGreaterEqual(len(route), 2)
        self.assertEqual(route[0], start_spot_index)
        self.assertEqual(route[-1], start_spot_index)


if __name__ == "__main__":
    unittest.main()
