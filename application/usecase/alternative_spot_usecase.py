"""
대체 여행지 추천 유스케이스 모듈
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from domain.entities.tourist_spot import TouristSpot
from domain.repositories.tourist_spot_repository import TouristSpotRepository
from domain.services.alternative_spot_service import AlternativeSpotService


@dataclass
class AlternativeSpotRequest:
    """대체 여행지 추천 요청 데이터"""

    itinerary: List[int]
    modify_idx: List[int]
    radius: float = 5.0
    recommend_count: int = 5


@dataclass
class AlternativeSpotResponse:
    """대체 여행지 추천 응답 데이터"""

    alternatives: Dict[str, List[int]]


class AlternativeSpotUseCase:
    """대체 여행지 추천 유스케이스"""

    def __init__(
        self,
        tourist_spot_repository: TouristSpotRepository,
        alternative_spot_service: AlternativeSpotService,
    ):
        """
        초기화

        Args:
            tourist_spot_repository: 관광지 저장소
            alternative_spot_service: 대체 여행지 추천 서비스
        """
        self.tourist_spot_repository = tourist_spot_repository
        self.alternative_spot_service = alternative_spot_service

    def execute(self, request: AlternativeSpotRequest) -> AlternativeSpotResponse:
        """
        유스케이스 실행

        Args:
            request: 요청 데이터

        Returns:
            응답 데이터
        """
        # 모든 관광지 조회 (실제 구현에서는 필요한 도시나 지역으로 필터링할 수 있음)
        all_spots = self.tourist_spot_repository.find_all()

        if not all_spots:
            raise ValueError("관광지 데이터를 찾을 수 없습니다.")

        # 대체 여행지 추천
        alternatives = self.alternative_spot_service.find_alternative_spots_multi(
            spots=all_spots,
            itinerary=request.itinerary,
            modify_idx=request.modify_idx,
            radius=request.radius,
            recommend_count=request.recommend_count
        )

        # 응답 생성
        return AlternativeSpotResponse(alternatives=alternatives)
