"""
대체 여행지 추천 서비스 모듈
"""

import random
from typing import Dict, List, Optional, Tuple

from domain.entities.tourist_spot import TouristSpot
from domain.value_objects.coordinate import Coordinate


class AlternativeSpotService:
    """대체 여행지 추천 서비스"""

    def find_alternative_spots_multi(
        self,
        spots: List[TouristSpot],
        itinerary: List[int],
        modify_idx: List[int],
        radius: float = 5.0,
        recommend_count: int = 5,
    ) -> Dict[str, List[int]]:
        """
        주어진 여행 일정에서 특정 인덱스의 관광지를 대체할 수 있는 추천 관광지 목록을 찾습니다.

        Args:
            spots: 전체 관광지 목록
            itinerary: 현재 여행 일정에 포함된 관광지 ID 목록
            modify_idx: 대체 여행지를 추천받을 인덱스 목록 (0부터 시작)
            radius: 대체 여행지 검색 반경 (km)
            recommend_count: 각 인덱스별 추천할 대체 여행지 개수

        Returns:
            인덱스별 대체 여행지 ID 목록을 담은 딕셔너리
        """
        # 관광지 ID를 키로 하는 관광지 사전 생성
        spot_dict: Dict[int, TouristSpot] = {
            spot.tourist_spot_id: spot for spot in spots
        }

        # 결과 딕셔너리 초기화
        result_dict: Dict[str, List[int]] = {}

        # 수정할 인덱스가 유효한지 확인
        valid_modify_idx = [idx for idx in modify_idx if 0 <= idx < len(itinerary)]

        # 각 수정 인덱스에 대해 대체 여행지 찾기
        for idx in valid_modify_idx:
            original_spot_id = itinerary[idx]

            # 원본 관광지가 spot_dict에 없으면 빈 리스트 반환
            if original_spot_id not in spot_dict:
                result_dict[str(idx)] = []
                continue

            original_spot = spot_dict[original_spot_id]

            # 반경 내 대체 관광지 찾기
            alternatives = self._find_spots_within_radius(
                original_spot,
                spots,
                radius,
                exclude_ids=itinerary,  # 현재 일정에 있는 관광지는 제외
            )

            # 대체 관광지 ID 목록 추출 (최대 recommend_count개)
            alternative_ids = [
                spot.tourist_spot_id for spot in alternatives[:recommend_count]
            ]
            result_dict[str(idx)] = alternative_ids

        return result_dict

    def find_alternative_spots(
        self,
        spots: List[TouristSpot],
        itinerary: List[int],
        modify_idx: List[int],
        radius: float = 5.0,
    ) -> List[int]:
        """
        주어진 여행 일정에서 특정 인덱스의 관광지를 대체할 수 있는 추천 관광지를 찾습니다.

        Args:
            spots: 전체 관광지 목록
            itinerary: 현재 여행 일정에 포함된 관광지 ID 목록
            modify_idx: 대체 여행지를 추천받을 인덱스 목록 (0부터 시작)
            radius: 대체 여행지 검색 반경 (km)

        Returns:
            대체 여행지가 포함된 새로운 여행 일정 관광지 ID 목록
        """
        # 관광지 ID를 키로 하는 관광지 사전 생성
        spot_dict: Dict[int, TouristSpot] = {
            spot.tourist_spot_id: spot for spot in spots
        }

        # 결과 여행 일정 초기화 (원본 복사)
        result_itinerary = itinerary.copy()

        # 수정할 인덱스가 유효한지 확인
        valid_modify_idx = [idx for idx in modify_idx if 0 <= idx < len(itinerary)]

        # 각 수정 인덱스에 대해 대체 여행지 찾기
        for idx in valid_modify_idx:
            original_spot_id = itinerary[idx]

            # 원본 관광지가 spot_dict에 없으면 건너뛰기
            if original_spot_id not in spot_dict:
                continue

            original_spot = spot_dict[original_spot_id]

            # 반경 내 대체 관광지 찾기
            alternatives = self._find_spots_within_radius(
                original_spot,
                spots,
                radius,
                exclude_ids=itinerary,  # 현재 일정에 있는 관광지는 제외
            )

            # 대체 관광지가 있으면 교체
            if alternatives:
                # 가장 적합한 대체 관광지 선택 (여기서는 간단히 첫 번째 항목 선택)
                # 실제 구현에서는 더 복잡한 선택 로직을 적용할 수 있음
                result_itinerary[idx] = alternatives[0].tourist_spot_id

        return result_itinerary

    def _find_spots_within_radius(
        self,
        center_spot: TouristSpot,
        all_spots: List[TouristSpot],
        radius: float,
        exclude_ids: List[int] = None,
    ) -> List[TouristSpot]:
        """
        중심 관광지로부터 지정된 반경 내에 있는 관광지를 찾습니다.

        Args:
            center_spot: 중심 관광지
            all_spots: 전체 관광지 목록
            radius: 검색 반경 (km)
            exclude_ids: 제외할 관광지 ID 목록

        Returns:
            반경 내 관광지 목록
        """
        if exclude_ids is None:
            exclude_ids = []

        # 중심 관광지의 좌표
        center_coord = center_spot.coordinate

        # 반경 내 관광지 찾기
        spots_within_radius = []

        for spot in all_spots:
            # 제외 목록에 있는 관광지는 건너뛰기
            if spot.tourist_spot_id in exclude_ids:
                continue

            # 중심 관광지와 동일한 관광지는 건너뛰기
            if spot.tourist_spot_id == center_spot.tourist_spot_id:
                continue

            # 두 관광지 간 거리 계산
            distance = center_coord.distance_to(spot.coordinate)

            # 반경 내에 있으면 목록에 추가
            if distance <= radius:
                spots_within_radius.append(spot)

        # 카테고리 유사성, 인기도 등을 고려하여 정렬 (향후 구현)
        # 여기서는 간단히 거리 기준으로 정렬
        spots_within_radius.sort(
            key=lambda spot: center_coord.distance_to(spot.coordinate)
        )

        return spots_within_radius
