"""
관광지 리포지토리 인터페이스 모듈
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities.tourist_spot import TouristSpot


class TouristSpotRepository(ABC):
    """관광지 정보에 접근하기 위한 리포지토리 인터페이스"""

    @abstractmethod
    def find_by_id(self, tourist_spot_id: int) -> Optional[TouristSpot]:
        """
        ID로 관광지 조회

        Args:
            tourist_spot_id: 관광지 ID

        Returns:
            관광지 객체 또는 None
        """
        pass

    @abstractmethod
    def find_by_city_id(self, city_id: int) -> List[TouristSpot]:
        """
        도시 ID로 관광지 목록 조회

        Args:
            city_id: 도시 ID

        Returns:
            관광지 객체 목록
        """
        pass

    @abstractmethod
    def find_by_ids(self, tourist_spot_ids: List[int]) -> List[TouristSpot]:
        """
        ID 목록으로 관광지 목록 조회

        Args:
            tourist_spot_ids: 관광지 ID 목록

        Returns:
            관광지 객체 목록
        """
        pass

    @abstractmethod
    def find_nearby(
        self, latitude: float, longitude: float, radius_km: float
    ) -> List[TouristSpot]:
        """
        주변 관광지 조회

        Args:
            latitude: 위도
            longitude: 경도
            radius_km: 반경 (km)

        Returns:
            관광지 객체 목록
        """
        pass

    @abstractmethod
    def find_by_category(self, categories: List[str]) -> List[TouristSpot]:
        """
        카테고리로 관광지 목록 조회

        Args:
            categories: 카테고리 목록

        Returns:
            관광지 객체 목록
        """
        pass

    @abstractmethod
    def save(self, tourist_spot: TouristSpot) -> TouristSpot:
        """
        관광지 저장

        Args:
            tourist_spot: 저장할 관광지 객체

        Returns:
            저장된 관광지 객체
        """
        pass
