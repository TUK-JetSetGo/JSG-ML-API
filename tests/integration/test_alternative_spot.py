"""
대체 여행지 추천 API 테스트 모듈 (다중 추천 결과)
"""

import unittest
from typing import Dict, List

from application.usecase.alternative_spot_usecase import (
    AlternativeSpotRequest, AlternativeSpotResponse, AlternativeSpotUseCase)
from domain.entities.tourist_spot import TouristSpot
from domain.services.alternative_spot_service import AlternativeSpotService
from domain.value_objects.coordinate import Coordinate


class MockTouristSpotRepository:
    """테스트용 관광지 저장소 Mock"""

    def find_all(self) -> List[TouristSpot]:
        """모든 관광지 조회"""
        return [
            TouristSpot(
                tourist_spot_id=123,
                name="관광지 1",
                coordinate=Coordinate(37.5665, 126.9780),
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=234,
                name="관광지 2",
                coordinate=Coordinate(37.5667, 126.9785),
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=345,
                name="관광지 3",
                coordinate=Coordinate(37.5670, 126.9790),
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=456,
                name="관광지 4",
                coordinate=Coordinate(37.5675, 126.9795),
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=567,
                name="대체 관광지 1-1",
                coordinate=Coordinate(37.5666, 126.9781),  # 관광지 1과 가까움
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=568,
                name="대체 관광지 1-2",
                coordinate=Coordinate(37.5667, 126.9782),  # 관광지 1과 가까움
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=678,
                name="대체 관광지 4-1",
                coordinate=Coordinate(37.5676, 126.9796),  # 관광지 4와 가까움
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=679,
                name="대체 관광지 4-2",
                coordinate=Coordinate(37.5677, 126.9797),  # 관광지 4와 가까움
                category=["명소"],
            ),
            TouristSpot(
                tourist_spot_id=789,
                name="먼 관광지",
                coordinate=Coordinate(37.6000, 127.0000),  # 모든 관광지와 멀리 떨어짐
                category=["명소"],
            ),
        ]


class TestAlternativeSpotServiceMulti(unittest.TestCase):
    """대체 여행지 추천 서비스 테스트 (다중 추천 결과)"""

    def setUp(self):
        """테스트 설정"""
        self.service = AlternativeSpotService()
        self.repository = MockTouristSpotRepository()
        self.spots = self.repository.find_all()

    def test_find_alternative_spots_multi(self):
        """다중 대체 여행지 추천 테스트"""
        # 테스트 데이터
        itinerary = [123, 234, 345, 456]
        modify_idx = [0, 3]
        radius = 0.5  # 500m 이내
        recommend_count = 2

        # 대체 여행지 추천
        result = self.service.find_alternative_spots_multi(
            spots=self.spots,
            itinerary=itinerary,
            modify_idx=modify_idx,
            radius=radius,
            recommend_count=recommend_count,
        )

        # 검증
        self.assertIn("0", result)  # 첫 번째 인덱스 결과 있음
        self.assertIn("3", result)  # 네 번째 인덱스 결과 있음
        self.assertEqual(len(result), 2)  # 두 개의 인덱스에 대한 결과

        # 각 인덱스별 추천 개수 확인
        self.assertEqual(len(result["0"]), 2)  # 첫 번째 인덱스에 2개 추천
        self.assertEqual(len(result["3"]), 2)  # 네 번째 인덱스에 2개 추천

        # 추천된 관광지 ID 확인
        self.assertIn(567, result["0"])  # 관광지 1의 대체로 대체 관광지 1-1 포함
        self.assertIn(568, result["0"])  # 관광지 1의 대체로 대체 관광지 1-2 포함
        self.assertIn(678, result["3"])  # 관광지 4의 대체로 대체 관광지 4-1 포함
        self.assertIn(679, result["3"])  # 관광지 4의 대체로 대체 관광지 4-2 포함

    def test_find_alternative_spots_multi_with_invalid_id(self):
        """존재하지 않는 관광지 ID에 대한 다중 대체 여행지 추천 테스트"""
        # 테스트 데이터 (999는 존재하지 않는 ID)
        itinerary = [123, 999, 345, 456]
        modify_idx = [0, 1, 3]
        radius = 0.5
        recommend_count = 2

        # 대체 여행지 추천
        result = self.service.find_alternative_spots_multi(
            spots=self.spots,
            itinerary=itinerary,
            modify_idx=modify_idx,
            radius=radius,
            recommend_count=recommend_count,
        )

        # 검증
        self.assertIn("0", result)  # 첫 번째 인덱스 결과 있음
        self.assertIn("1", result)  # 두 번째 인덱스 결과 있음 (존재하지 않는 ID)
        self.assertIn("3", result)  # 네 번째 인덱스 결과 있음

        # 존재하지 않는 ID에 대해서는 빈 리스트 반환
        self.assertEqual(result["1"], [])

        # 다른 인덱스는 정상적으로 추천
        self.assertEqual(len(result["0"]), 2)
        self.assertEqual(len(result["3"]), 2)


class TestAlternativeSpotUseCaseMulti(unittest.TestCase):
    """대체 여행지 추천 유스케이스 테스트 (다중 추천 결과)"""

    def setUp(self):
        """테스트 설정"""
        self.repository = MockTouristSpotRepository()
        self.service = AlternativeSpotService()
        self.usecase = AlternativeSpotUseCase(
            tourist_spot_repository=self.repository,
            alternative_spot_service=self.service,
        )

    def test_execute_multi(self):
        """다중 추천 유스케이스 실행 테스트"""
        # 테스트 데이터
        request = AlternativeSpotRequest(
            itinerary=[123, 234, 345, 456],
            modify_idx=[0, 3],
            radius=0.5,
            recommend_count=2,
        )

        # 유스케이스 실행
        response = self.usecase.execute(request)

        # 검증
        self.assertIsInstance(response, AlternativeSpotResponse)
        self.assertIn("0", response.alternatives)
        self.assertIn("3", response.alternatives)
        self.assertEqual(len(response.alternatives["0"]), 2)
        self.assertEqual(len(response.alternatives["3"]), 2)


if __name__ == "__main__":
    unittest.main()
