import math
import unittest
from typing import Dict, List

from infrastructure.adapters.ml.enhanced_ptppp_model import EnhancedPTPPPModel


class DummyCoordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
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
        self.id = tourist_spot_id


class DummyUserProfile:
    def __init__(self, must_visit_list: List[int] = None):
        self.must_visit_list = must_visit_list or []


class TestEnhancedPTPPPModel(unittest.TestCase):

    def setUp(self):
        self.spots = [
            DummyTouristSpot(1, 37.0, 127.0, average_visit_duration=0.5),
            DummyTouristSpot(2, 37.01, 127.01, average_visit_duration=0.75),
            DummyTouristSpot(3, 37.02, 127.02, average_visit_duration=0.6),
            DummyTouristSpot(4, 37.03, 127.03, average_visit_duration=0.8),
        ]
        self.user_profile = DummyUserProfile(must_visit_list=[2])
        self.base_scores: Dict[int, float] = {1: 5.0, 2: 7.0, 3: 6.0, 4: 8.0}
        self.priority_scores: Dict[int, Dict[int, float]] = {
            1: {1: 1.0, 2: 2.0, 3: 1.5, 4: 2.5},
            2: {1: 0.5, 2: 1.0, 3: 0.8, 4: 1.2},
            3: {1: 0.3, 2: 0.7, 3: 0.5, 4: 0.9},
        }
        self.model = EnhancedPTPPPModel()

    def test_calculate_distance_matrix(self):
        matrix = self.model.calculate_distance_matrix(self.spots)
        n = len(self.spots)
        self.assertEqual(len(matrix), n)
        for row in matrix:
            self.assertEqual(len(row), n)
        for i in range(n):
            self.assertAlmostEqual(matrix[i][i], 0.0)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertGreater(matrix[i][j], 0.0)

    def test_optimize_route(self):
        start_spot_index = 0  # 시작은 첫 번째 관광지
        max_distance = 50.0  # km 단위 (임의 값)
        max_duration = 5.0  # 시간 단위 (임의 값)
        max_places = 4  # 방문할 수 있는 관광지 수 (시작 지점 제외)
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
        self.assertIsInstance(route, list)
        self.assertGreaterEqual(total_distance, 0.0)
        self.assertGreaterEqual(total_duration, 0.0)
        if len(route) > 0:
            self.assertIn(1, route)

    def test_optimize_multi_day_route(self):
        daily_spots: List[List[DummyTouristSpot]] = [self.spots, self.spots]
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
        self.assertIsInstance(multi_day_routes, list)
        self.assertEqual(len(multi_day_routes), len(daily_spots))
        for daily_route in multi_day_routes:
            route_indices, day_distance, day_duration = daily_route
            self.assertIsInstance(route_indices, list)
            self.assertGreaterEqual(day_distance, 0.0)
            self.assertGreaterEqual(day_duration, 0.0)

    def test_get_must_visit_indices(self):
        must_visit = self.model._get_must_visit_indices(self.spots, self.user_profile)
        self.assertIn(1, must_visit)

    def test_optimize_single_day_route_without_must_visit(self):
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


if __name__ == "__main__":
    unittest.main()
