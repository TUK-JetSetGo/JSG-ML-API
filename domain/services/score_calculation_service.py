from typing import Dict, List, Optional

import numpy as np

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.value_objects.category import CategorySet


class ScoreCalculationService:
    """관광지 점수 계산을 위한 도메인 서비스"""

    @staticmethod
    def calculate_base_score(
        spot: TouristSpot,
        user_profile: Optional[UserProfile] = None,
        base_scale: float = 1.0,
    ) -> float:
        """
        관광지의 기본 점수 계산

        Args:
            spot: 관광지
            user_profile: 사용자 프로필
            base_scale: 기본 점수 스케일 계수

        Returns:
            기본 점수
        """

        base_score = spot.activity_level * base_scale

        if user_profile:
            category_score = ScoreCalculationService._calculate_category_math_score(
                spot, user_profile
            )
            base_score += category_score

            if spot.id in user_profile.must_visit_list:
                base_score += 1000.0

            if spot.id in user_profile.not_visit_list:
                base_score = -1000.0

        return base_score

    @staticmethod
    def _calculate_category_math_score(
        spot: TouristSpot, user_profile: UserProfile
    ) -> float:
        """
        관광지와 사용자 프로필 간의 카테고리 매칭 점수 계산

        Args:
            spot: 관광지
            user_profile: 사용자 프로필

        Returns:
            카테고리 매칭 점수
        """
        if not spot.category or not user_profile.themes:
            return 0.0

        spot_categories = CategorySet()
        spot_categories.add_from_list(spot.category)

        user_categories = CategorySet()
        user_categories.add_from_list(user_profile.themes)

        match_score = spot_categories.match_score(user_categories)

        return match_score * 100.0

    @staticmethod
    def _calculate_category_match_score(
        spot: TouristSpot, user_profile: UserProfile
    ) -> float:
        """
        관광지와 사용자 프로필 간의 카테고리 매칭 점수 계산

        Args:
            spot: 관광지
            user_profile: 사용자 프로필

        Returns:
            카테고리 매칭 점수
        """

        if not spot.category or not user_profile.themes:
            return 0.0

        spot_categories = CategorySet()
        spot_categories.add_from_list(spot.category)
        user_categories = CategorySet()
        user_categories.add_from_list(user_profile.themes)
        match_score = spot_categories.match_score(user_categories)
        return match_score * 100.0

    @staticmethod
    def calculate_priority_scores(
        spots: List[TouristSpot],
        user_profile: Optional[UserProfile] = None,
        base_scores: Optional[Dict[int, float]] = None,
        priority_scale: float = 0.3,
        max_priority: int = 10,
    ) -> Dict[int, Dict[int, float]]:
        """
        관광지의 우선순위별 점수 계산

        Args:
            spots: 관광지 목록
            user_profile: 사용자 프로필
            base_scores: 미리 계산된 기본 점수
            priority_scale: 우선순위 스케일 계수
            max_priority: 최대 우선순위 레벨

        Returns:
            우선순위 레벨을 첫 번째 키로, 관광지 ID를 두 번째 키로, 점수를 값으로 하는 중첩 딕셔너리
        """
        if base_scores is None:
            base_scores = {
                spot.tourist_spot_id: ScoreCalculationService.calculate_base_score(
                    spot, user_profile
                )
                for spot in spots
            }
        priority_scores: Dict[int, Dict[int, float]] = {}
        for k in range(1, max_priority + 1):
            priority_scores[k] = {}
            for spot in spots:
                base_scores = base_scores.get(spot.tourist_spot_id, 0.0)
                priority_scores[k][spot.tourist_spot_id] = (
                    base_scores * priority_scale / np.sqrt(k)
                )

        return priority_scores

    @staticmethod
    def calculate_time_compatibility_score(
        spot: TouristSpot, current_time_hours: float, max_score: float = 10.0
    ) -> float:
        """
        현재 시간과 관광지의 호환성 점수 계산

        Args:
            spot: 관광지
            current_time_hours: 현재 시간 (0.0 ~ 24.0)
            max_score: 최대 점수

        Returns:
            시간 호환성 점수
        """
        hours = int(current_time_hours)
        minutes = int((current_time_hours - hours) * 60)
        current_time = f"{hours:02d}:{minutes:02d}"

        opening_time_str = spot.opening_time.strftime("%H:%M")
        closing_time_str = spot.closing_time.strftime("%H:%M")

        if opening_time_str <= current_time <= closing_time_str:
            return max_score
        if current_time < opening_time_str:
            opening_hours = int(opening_time_str.split(":")[0])
            opening_minutes = int(opening_time_str.split(":")[1])
            opening_time_value = opening_hours + opening_minutes / 60.0
            time_diff = opening_time_value - current_time_hours
            normalized_diff = max(0, 1 - time_diff / 6.0)
            return max_score * normalized_diff
        else:
            closing_hours = int(closing_time_str.split(":")[0])
            closing_minutes = int(closing_time_str.split(":")[1])
            closing_time_value = closing_hours + closing_minutes / 60.0
            time_diff = current_time_hours - closing_time_value
            normalized_diff = max(0, 1 - time_diff / 2.0)
            return max_score * normalized_diff

    @staticmethod
    def calculate_popularity_score(spot: TouristSpot) -> float:
        """
        관광지의 인기도 점수를 계산합니다.

        activity_level을 10점 만점으로 정규화하여 반환합니다.
        (activity_level은 0~10 범위)
        """
        return float(spot.activity_level)

    @staticmethod
    def calculate_theme_match_score(
        spot: TouristSpot, user_profile: UserProfile
    ) -> float:
        """
        관광지의 테마 일치도 점수를 계산합니다.

        내부의 _calculate_category_match_score를 이용해 0~100 범위의 점수를 얻은 후,
        이를 0~10 범위로 변환하여 반환합니다.
        """
        base_theme_score = ScoreCalculationService._calculate_category_match_score(
            spot, user_profile
        )
        return base_theme_score / 10.0
