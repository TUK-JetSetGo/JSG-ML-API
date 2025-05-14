"""
Tabu Search를 이용한 경로 최적화 서비스입니다.

이 서비스는 `EnhancedTabuSearchModel`을 사용하여 단일일 또는 다중일 여정에 대한
최적화된 경로를 생성하는 통합 인터페이스를 제공합니다.
사용자 프로필에 기반한 이동 속도 결정 및 경로 연속성 부여는 모델 내부에서 처리됩니다.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from infrastructure.adapters.ml.enhanced_tabu_search_model import \
    EnhancedTabuSearchModel

logger = logging.getLogger(__name__)


class RouteOptimizationService:
    def __init__(self):
        """서비스 초기화 시 최적화 모델 인스턴스를 생성합니다."""
        self.model = EnhancedTabuSearchModel()

    @staticmethod
    def calculate_distance_matrix(spots: List[TouristSpot]) -> List[List[float]]:
        """
        관광지 간의 거리 행렬(km 단위)을 계산합니다. (모델 내부 계산 로직 사용 권장)
        """
        n = len(spots)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                matrix[i][j] = spots[i].coordinate.distance_to(spots[j].coordinate)
        return matrix

    def generate_itinerary(
        self,
        daily_spots_input: Union[List[TouristSpot], List[List[TouristSpot]]],
        user_profile: UserProfile,
        daily_start_indices_input: Union[int, List[int]],
        daily_max_distance_input: Union[float, List[float]],
        daily_max_duration_input: Union[float, List[float]],
        max_places_per_day: int,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],  # {order: {spot_id: score}}
        must_visit_spot_ids_input: Optional[Union[Set[int], List[Set[int]]]] = None,
        ts_max_iterations_per_day: int = 200,
    ) -> List[Tuple[List[int], float, float]]:
        """
        단일일 또는 다중일 최적화 여정을 생성합니다.
        """
        is_multi_day_request = (
            isinstance(daily_spots_input, list)
            and bool(daily_spots_input)
            and isinstance(daily_spots_input[0], list)
        )

        logger.info(
            f"경로 최적화 서비스 호출됨. 요청 유형: {'다중일' if is_multi_day_request else '단일일'}"
        )

        if not daily_spots_input:
            logger.warning(
                "입력된 일일 관광지 정보(daily_spots_input)가 비어있습니다. 빈 결과를 반환합니다."
            )
            return []

        if is_multi_day_request:
            num_days = len(daily_spots_input)
            if not (
                isinstance(daily_start_indices_input, list)
                and len(daily_start_indices_input) == num_days
            ):
                logger.error(
                    f"데이터 불일치: {num_days}일 일정의 경우, `daily_start_indices_input`은 길이가 {num_days}인 리스트여야 합니다. 입력: {daily_start_indices_input}"
                )
                return []
            if must_visit_spot_ids_input is not None and not (
                isinstance(must_visit_spot_ids_input, list)
                and len(must_visit_spot_ids_input) == num_days
            ):
                logger.error(
                    f"데이터 불일치: {num_days}일 일정의 경우, `must_visit_spot_ids_input`은 길이가 {num_days}인 리스트여야 합니다. 입력: {must_visit_spot_ids_input}"
                )
                return []
        else:  # 단일일 요청
            if not isinstance(daily_start_indices_input, int):
                logger.error(
                    f"데이터 불일치: 단일일 일정의 경우, `daily_start_indices_input`은 정수여야 합니다. 입력: {daily_start_indices_input}"
                )
                return []
            if must_visit_spot_ids_input is not None and not isinstance(
                must_visit_spot_ids_input, set
            ):
                logger.error(
                    f"데이터 불일치: 단일일 일정의 경우, `must_visit_spot_ids_input`은 Set[int]여야 합니다. 입력: {must_visit_spot_ids_input}"
                )
                return []

        try:
            optimized_routes = self.model.optimize_route(
                daily_spots_input=daily_spots_input,
                user_profile=user_profile,
                daily_start_indices_input=daily_start_indices_input,
                daily_max_distance_input=daily_max_distance_input,
                daily_max_duration_input=daily_max_duration_input,
                max_places_per_day=max_places_per_day,
                base_scores=base_scores,
                priority_scores=priority_scores,
                must_visit_spot_ids_input=must_visit_spot_ids_input,
                ts_max_iterations_per_day=ts_max_iterations_per_day,
            )
            logger.info(f"경로 최적화 완료. 결과 일 수: {len(optimized_routes)}")
            return optimized_routes
        except ValueError as e:
            logger.error(f"모델 내부 경로 최적화 중 ValueError 발생: {e}")
            return []
        except Exception as e:
            logger.error(f"경로 최적화 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return []

    def optimize_route(
        self,
        spots: List[TouristSpot],
        start_spot_index: int,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        max_distance: float,
        max_duration: float,
        max_places: int,
        must_visit_indices: List[int],
        transport_speed_kmh: float = 40.0,
        user_profile_override: Optional[UserProfile] = None,
    ) -> Tuple[List[int], float, float]:
        """
        기존 optimize_route 인터페이스와의 하위 호환성을 위한 래퍼 메서드입니다.
        단일일 경로 최적화를 수행합니다.
        """
        logger.info(
            f"optimize_route (호환성 래퍼) 호출됨. spots: {len(spots)}, start_idx: {start_spot_index}"
        )
        if not spots:
            logger.warning(
                "optimize_route: 입력된 관광지(spots)가 비어있습니다. 빈 결과를 반환합니다."
            )
            return [], 0.0, 0.0

        if user_profile_override:
            current_user_profile = user_profile_override
        else:
            current_user_profile = UserProfile(
                travel_type="solo",
                group_size=1,
                budget_amount=100000,
                themes=[],
                preferred_transport="default",
                must_visit_list=[],
                not_visit_list=[],
            )

        must_visit_spot_ids: Set[int] = set()
        if spots and must_visit_indices:
            for idx in must_visit_indices:
                if 0 <= idx < len(spots):
                    spot_id = getattr(
                        spots[idx], "tourist_spot_id", getattr(spots[idx], "id", None)
                    )
                    if spot_id is not None:
                        must_visit_spot_ids.add(spot_id)
                    else:
                        logger.warning(
                            f"관광지 ID를 찾을 수 없습니다: spots[{idx}] (must_visit_indices)"
                        )
                else:
                    logger.warning(f"잘못된 must_visit_index 발견: {idx}")

        results = self.generate_itinerary(
            daily_spots_input=spots,
            user_profile=current_user_profile,
            daily_start_indices_input=start_spot_index,
            daily_max_distance_input=max_distance,
            daily_max_duration_input=max_duration,
            max_places_per_day=max_places,
            base_scores=base_scores,
            priority_scores=priority_scores,
            must_visit_spot_ids_input=must_visit_spot_ids,
        )

        if (
            results
            and isinstance(results, list)
            and len(results) > 0
            and isinstance(results[0], tuple)
            and len(results[0]) == 3
        ):
            return results[0]
        else:
            logger.warning(
                "optimize_route: generate_itinerary로부터 유효한 결과를 받지 못했습니다."
            )
            return [], 0.0, 0.0
