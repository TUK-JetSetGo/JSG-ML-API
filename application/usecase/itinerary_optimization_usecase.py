"""
여행 일정 최적화 유스케이스 모듈
"""

import datetime
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from domain.entities.itinerary import Itinerary, ItineraryDay, ItineraryItem
from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.repositories.tourist_spot_repository import TouristSpotRepository
from domain.services.clustering_service import ClusteringService
from domain.services.route_optimization_service import RouteOptimizationService
from domain.services.score_calculation_service import ScoreCalculationService
from domain.services.time_calculation_service import TimeCalculationService
from infrastructure.adapters.ml.enhanced_clustering_algorithm import \
    EnhancedClusteringAlgorithm
from infrastructure.adapters.ml.user_preference_model import \
    UserPreferenceModel

logger = logging.getLogger(__name__)


@dataclass
class ItineraryOptimizationRequest:
    """여행 일정 최적화 요청 데이터

    레거시 코드처럼 user_id를 포함하지만,
    사용자 정보는 UserProfile 객체로 직접 입력받습니다.
    (user_repository 미사용)
    """

    user_id: int
    city_id: int
    num_days: int
    user_profile: UserProfile

    start_date: Optional[datetime.date] = None
    max_spots_per_day: int = 5
    max_distance_per_day_km: float = 150.0
    max_duration_per_day_hours: float = 8.0
    transport_speed_kmh: float = 40.0
    optimize_for_theme: bool = True
    optimize_for_distance: bool = True
    optimize_for_time: bool = True
    optimize_for_popularity: bool = True
    optimize_for_user_preference: bool = True
    optimize_for_day_connectivity: bool = True


@dataclass
class ItineraryOptimizationResponse:
    """여행 일정 최적화 응답 데이터"""

    itinerary: Dict[str, Any]
    user_profile: Dict[str, Any]
    stats: Dict[str, Any]
    daily_routes: List[Dict[str, Any]]


class ItineraryOptimizationUseCase:
    """여행 일정 최적화 유스케이스"""

    def __init__(
            self,
            tourist_spot_repository: TouristSpotRepository,
            clustering_service: ClusteringService,
            route_optimization_service: RouteOptimizationService,
            score_calculation_service: ScoreCalculationService,
            time_calculation_service: TimeCalculationService,
            user_preference_model: UserPreferenceModel,
            enhanced_clustering_algorithm: EnhancedClusteringAlgorithm,
    ):
        """
        초기화

        user_repository는 전혀 사용하지 않습니다.
        """
        self.tourist_spot_repository = tourist_spot_repository
        self.clustering_service = clustering_service
        self.route_optimization_service = route_optimization_service
        self.score_calculation_service = score_calculation_service
        self.time_calculation_service = time_calculation_service
        self.user_preference_model = user_preference_model
        self.enhanced_clustering_algorithm = enhanced_clustering_algorithm

    def execute(
            self, request: ItineraryOptimizationRequest
    ) -> ItineraryOptimizationResponse:
        """
        유스케이스 실행

        Args:
            request: 요청 데이터

        Returns:
            응답 데이터
        """

        user_profile = request.user_profile
        logger.info(f"[itinerary_optimization_usecase.execute] step1 finished")

        spots = self.tourist_spot_repository.find_by_city_id(request.city_id)
        if not spots:
            raise ValueError(f"도시에 관광지가 없습니다: {request.city_id}")
        logger.info(f"[itinerary_optimization_usecase.execute] step2 finished")

        if not hasattr(self.user_preference_model.user_features_scaler, "mean_"):
            dummy_users = generate_dummy_users(num_users=100)
            dummy_ratings = generate_dummy_ratings(dummy_users, spots)
            self.user_preference_model.train(dummy_users, spots, dummy_ratings)
        logger.info(f"[itinerary_optimization_usecase.execute] step3 finished")

        spot_scores = {}
        for spot in spots:
            # 기본 점수
            base_score = self.score_calculation_service.calculate_base_score(spot)

            if request.optimize_for_user_preference:
                user_preference_score = self.user_preference_model.predict_preference(
                    user_profile, spot
                )
            else:
                user_preference_score = 0.0

            # 인기도 점수
            if request.optimize_for_popularity:
                popularity_score = (
                    self.score_calculation_service.calculate_popularity_score(spot)
                )
            else:
                popularity_score = 0.0

            # 테마 일치도 점수
            if request.optimize_for_theme:
                theme_score = (
                    self.score_calculation_service.calculate_theme_match_score(
                        spot, user_profile
                    )
                )
            else:
                theme_score = 0.0

            # 종합 점수
            total_score = (
                    base_score * 0.2
                    + user_preference_score * 0.3
                    + popularity_score * 0.2
                    + theme_score * 0.3
            )
            if spot.tourist_spot_id in user_profile.must_visit_list:
                total_score = max(total_score, 9.0)

            if spot.tourist_spot_id in user_profile.not_visit_list:
                total_score = min(total_score, 1.0)

            spot_scores[spot.tourist_spot_id] = total_score

        logger.info(f"[itinerary_optimization_usecase.execute] step4 finished")

        # 5. 클러스터링
        clusters = self.enhanced_clustering_algorithm.cluster_spots_for_multi_day_trip(
            spots=spots,
            user_profile=user_profile,
            num_days=request.num_days,
            base_scores=spot_scores,
        )
        logger.info(f"[itinerary_optimization_usecase.execute] step5 finished")

        # 6. 일자별 최적 클러스터 선택
        selected_clusters = self.enhanced_clustering_algorithm.select_optimal_clusters(
            clusters=clusters,
            user_profile=user_profile,
            num_days=request.num_days,
            base_scores=spot_scores,
        )
        logger.info(f"[itinerary_optimization_usecase.execute] step6 finished")

        # 7. 일자별 경로 최적화
        daily_routes = []
        daily_stats = []

        for day_idx, cluster_spots in enumerate(selected_clusters):
            logger.info(
                f"[step7] day {day_idx + 1}: cluster_spots count={len(cluster_spots)}, selected_clusters total={len(selected_clusters)}"
            )
            if day_idx == 0 or not daily_routes:
                start_spot_idx = 0
            else:
                prev_day_route = daily_routes[-1][0]
                if prev_day_route and len(prev_day_route) >= 2:
                    start_spot_idx = prev_day_route[-2]
                else:
                    start_spot_idx = 0

            if cluster_spots:
                safe_start = min(max(start_spot_idx, 0), len(cluster_spots) - 1)
                if safe_start != start_spot_idx:
                    logger.warning(
                        f"[step7] day {day_idx + 1}: adjusted start_spot_idx from {start_spot_idx} to {safe_start}"
                    )
                start_spot_idx = safe_start
            else:
                start_spot_idx = 0
            # 일자별 우선순위 점수
            priority_scores = {
                spot.tourist_spot_id: spot_scores.get(spot.tourist_spot_id, 0.0)
                for spot in cluster_spots
            }

            # 반드시 방문해야 하는 지점 인덱스
            must_visit_indices = []
            for i, spot in enumerate(cluster_spots):
                if spot.tourist_spot_id in user_profile.must_visit_list:
                    must_visit_indices.append(i)

            # 경로 최적화
            route, total_dist, total_dur = (
                self.route_optimization_service.optimize_route(
                    spots=cluster_spots,
                    start_spot_index=start_spot_idx,
                    base_scores=spot_scores,
                    priority_scores={day_idx: priority_scores},
                    max_distance=request.max_distance_per_day_km,
                    max_duration=request.max_duration_per_day_hours,
                    max_places=request.max_spots_per_day + 1,
                    must_visit_indices=must_visit_indices,
                    transport_speed_kmh=request.transport_speed_kmh,
                )
            )
            logger.info(
                f"[step7] day {day_idx + 1}: optimized route={route}, dist={total_dist}, dur={total_dur}"
            )

            logger.info(
                f"[step7] day {day_idx + 1}: pre-fallback check, route length={len(route)}"
            )
            if any(idx < 0 or idx >= len(cluster_spots) for idx in route):
                logger.warning(
                    f"[itinerary_optimization_usecase] Invalid route indices for day {day_idx + 1}: {route}"
                )

                if cluster_spots:
                    safe_start_idx_in_cluster = -1
                    if cluster_spots:
                        safe_start_idx_in_cluster = min(max(start_spot_idx, 0), len(cluster_spots) - 1)

                    max_fallback_spots = request.max_spots_per_day
                    current_fallback_route_indices = []

                    if cluster_spots and 0 <= safe_start_idx_in_cluster < len(cluster_spots):
                        current_fallback_route_indices.append(safe_start_idx_in_cluster)

                    if cluster_spots:
                        for spot_idx_in_cluster in must_visit_indices:
                            if len(current_fallback_route_indices) < max_fallback_spots:
                                if 0 <= spot_idx_in_cluster < len(
                                        cluster_spots) and spot_idx_in_cluster not in current_fallback_route_indices:
                                    current_fallback_route_indices.append(spot_idx_in_cluster)
                            else:
                                break

                    if cluster_spots and len(current_fallback_route_indices) < max_fallback_spots:

                        available_spots_for_scoring = []
                        for i_spot_idx in range(len(cluster_spots)):
                            if i_spot_idx not in current_fallback_route_indices:
                                score = priority_scores.get(cluster_spots[i_spot_idx].tourist_spot_id, 0.0)
                                available_spots_for_scoring.append((score, i_spot_idx))

                        available_spots_sorted_by_score = sorted(
                            available_spots_for_scoring,
                            key=lambda x: x[0],
                            reverse=True,
                        )

                        for _score, spot_idx_in_cluster in available_spots_sorted_by_score:
                            if len(current_fallback_route_indices) >= max_fallback_spots:
                                break
                            if spot_idx_in_cluster in current_fallback_route_indices:
                                continue
                            temp_route = current_fallback_route_indices + [spot_idx_in_cluster]
                            temp_timetable = self.time_calculation_service.generate_timetable(
                                spots=cluster_spots,
                                route=temp_route,
                            )
                            temp_travel_dur = sum(
                                item["duration"] for item in temp_timetable if item["type"] == "travel")
                            temp_travel_dist = temp_travel_dur * request.transport_speed_kmh
                            if temp_travel_dist <= request.max_distance_per_day_km:
                                current_fallback_route_indices.append(spot_idx_in_cluster)

                    fallback_route = current_fallback_route_indices
                else:
                    fallback_route = []
                logger.warning(
                    f"[step7] day {day_idx + 1}: invalid indices, fallback_route={fallback_route}"
                )
                timetable = self.time_calculation_service.generate_timetable(
                    spots=cluster_spots,
                    route=fallback_route,
                )
                fallback_dur = sum(
                    item["duration"] for item in timetable if item["type"] == "travel"
                )
                fallback_dist = fallback_dur * request.transport_speed_kmh
                daily_routes.append((fallback_route, fallback_dist, fallback_dur))
                daily_stats.append(
                    {
                        "day": day_idx + 1,
                        "num_spots": len(fallback_route),
                        "total_distance_km": fallback_dist,
                        "total_duration_hours": fallback_dur,
                        "spots": [
                            cluster_spots[i].tourist_spot_id for i in fallback_route if 0 <= i < len(cluster_spots)
                        ],
                    }
                )
                continue
            if not route or len(route) < 2:
                if cluster_spots:
                    safe_start_idx_in_cluster = -1
                    if cluster_spots:
                        safe_start_idx_in_cluster = min(max(start_spot_idx, 0), len(cluster_spots) - 1)

                    max_fallback_spots = request.max_spots_per_day
                    current_fallback_route_indices = []

                    if cluster_spots and 0 <= safe_start_idx_in_cluster < len(cluster_spots):
                        current_fallback_route_indices.append(safe_start_idx_in_cluster)

                    if cluster_spots:
                        for spot_idx_in_cluster in must_visit_indices:
                            if len(current_fallback_route_indices) < max_fallback_spots:
                                if 0 <= spot_idx_in_cluster < len(
                                        cluster_spots) and spot_idx_in_cluster not in current_fallback_route_indices:
                                    current_fallback_route_indices.append(spot_idx_in_cluster)
                            else:
                                break

                    if cluster_spots and len(current_fallback_route_indices) < max_fallback_spots:
                        available_spots_for_scoring = []
                        for i_spot_idx in range(len(cluster_spots)):
                            if i_spot_idx not in current_fallback_route_indices:
                                score = priority_scores.get(cluster_spots[i_spot_idx].tourist_spot_id, 0.0)
                                available_spots_for_scoring.append((score, i_spot_idx))

                        available_spots_sorted_by_score = sorted(
                            available_spots_for_scoring,
                            key=lambda x: x[0],
                            reverse=True,
                        )

                        for _score, spot_idx_in_cluster in available_spots_sorted_by_score:
                            if len(current_fallback_route_indices) >= max_fallback_spots:
                                break
                            if spot_idx_in_cluster in current_fallback_route_indices:
                                continue

                            temp_route = current_fallback_route_indices + [spot_idx_in_cluster]
                            temp_timetable = self.time_calculation_service.generate_timetable(
                                spots=cluster_spots,
                                route=temp_route,
                            )
                            temp_travel_dur = sum(
                                item["duration"] for item in temp_timetable if item["type"] == "travel")
                            temp_travel_dist = temp_travel_dur * request.transport_speed_kmh
                            if temp_travel_dist <= request.max_distance_per_day_km:
                                current_fallback_route_indices.append(spot_idx_in_cluster)

                    fallback_route = current_fallback_route_indices
                else:
                    fallback_route = []
                logger.warning(
                    f"[step7] day {day_idx + 1}: empty or too short route, fallback_route={fallback_route}"
                )
                timetable = self.time_calculation_service.generate_timetable(
                    spots=cluster_spots,
                    route=fallback_route,
                )
                fallback_dur = sum(
                    item["duration"] for item in timetable if item["type"] == "travel"
                )
                fallback_dist = fallback_dur * request.transport_speed_kmh
                daily_routes.append((fallback_route, fallback_dist, fallback_dur))
                daily_stats.append(
                    {
                        "day": day_idx + 1,
                        "num_spots": len(fallback_route),
                        "total_distance_km": fallback_dist,
                        "total_duration_hours": fallback_dur,
                        "spots": [
                            cluster_spots[i].tourist_spot_id for i in fallback_route if 0 <= i < len(cluster_spots)
                        ],
                    }
                )
                continue

            daily_routes.append((route, total_dist, total_dur))
            logger.info(f"[step7] day {day_idx + 1}: daily_routes now={daily_routes}")

            # 일자별 통계
            daily_stats.append(
                {
                    "day": day_idx + 1,
                    "num_spots": len(route) - 2,
                    "total_distance_km": total_dist,
                    "total_duration_hours": total_dur,
                    "spots": [
                        cluster_spots[idx].tourist_spot_id for idx in route[1:-1]
                    ],
                }
            )
        logger.info(
            f"[itinerary_optimization_usecase] Collected daily_stats: {daily_stats}"
        )
        logger.info(f"[itinerary_optimization_usecase.execute] step7 finished")
        # 8. 일정표 생성
        itinerary = Itinerary(
            start_date=request.start_date,
            end_date=(
                request.start_date + datetime.timedelta(days=request.num_days - 1)
                if request.start_date
                else None
            ),
            days=[],
        )

        for day_idx, (route, total_dist, total_dur) in enumerate(daily_routes):
            logger.info(
                f"[step8] day {day_idx + 1}: preparing timetable, day_spots count={len(selected_clusters[day_idx])}, route={route}"
            )
            day_spots = selected_clusters[day_idx]
            if not route or not day_spots:
                day_obj = ItineraryDay(day_number=day_idx + 1, items=[])
                itinerary.days.append(day_obj)
                continue
            timetable = self.time_calculation_service.generate_timetable(
                spots=day_spots, route=route
            )

            day_obj = ItineraryDay(day_number=day_idx + 1, items=[])

            for item_idx, timetable_item in enumerate(timetable):
                item_type = timetable_item["type"]
                if item_type == "visit":
                    item = ItineraryItem(
                        item_number=item_idx + 1,
                        type="visit",
                        spot_id=timetable_item["spot_id"],
                        spot_name=timetable_item["spot_name"],
                        start_time=timetable_item["start_time"],
                        end_time=timetable_item["end_time"],
                        duration=timetable_item["duration"],
                    )
                else:
                    item = ItineraryItem(
                        item_number=item_idx + 1,
                        type="travel",
                        from_spot_id=timetable_item["from_spot_id"],
                        from_spot_name=timetable_item["from_spot_name"],
                        to_spot_id=timetable_item["to_spot_id"],
                        to_spot_name=timetable_item["to_spot_name"],
                        start_time=timetable_item["start_time"],
                        end_time=timetable_item["end_time"],
                        duration=timetable_item["duration"],
                    )
                day_obj.items.append(item)

            itinerary.days.append(day_obj)
        logger.info(f"[itinerary_optimization_usecase.execute] step8 finished")
        saved_itinerary = itinerary
        logger.info(f"[itinerary_optimization_usecase.execute] step9 finished")
        # 10. 응답 데이터 구성
        #     레거시처럼 user_id를 명시하고,
        #     user_profile는 딕셔너리로 만들어 돌려줍니다.
        itinerary_dict = {
            "start_date": (
                saved_itinerary.start_date.isoformat()
                if saved_itinerary.start_date
                else None
            ),
            "end_date": (
                saved_itinerary.end_date.isoformat()
                if saved_itinerary.end_date
                else None
            ),
            "days": [],
        }
        logger.info(
            f"[itinerary_optimization_usecase] itinerary_dict: {itinerary_dict}"
        )

        # 일자 정보
        for day in saved_itinerary.days:
            day_dict = {"day_number": day.day_number, "items": []}
            for item in day.items:
                item_dict = {
                    "item_number": item.item_number,
                    "type": item.type,
                    "start_time": (
                        item.start_time.strftime("%H:%M") if item.start_time else None
                    ),
                    "end_time": (
                        item.end_time.strftime("%H:%M") if item.end_time else None
                    ),
                    "duration": item.duration,
                }
                if item.type == "visit":
                    item_dict.update(
                        {"spot_id": item.spot_id, "spot_name": item.spot_name}
                    )
                else:
                    item_dict.update(
                        {
                            "from_spot_id": item.from_spot_id,
                            "from_spot_name": item.from_spot_name,
                            "to_spot_id": item.to_spot_id,
                            "to_spot_name": item.to_spot_name,
                        }
                    )
                day_dict["items"].append(item_dict)
            itinerary_dict["days"].append(day_dict)

        user_profile_dict = {
            "travel_type": user_profile.travel_type,
            "group_size": user_profile.group_size,
            "budget_amount": user_profile.budget_amount,
            "themes": user_profile.themes,
            "must_visit_list": user_profile.must_visit_list,
            "not_visit_list": user_profile.not_visit_list,
            "preferred_transport": user_profile.preferred_transport,
            "preferred_activity_level": user_profile.preferred_activity_level,
            "preferred_start_time": user_profile.preferred_start_time,
            "preferred_end_time": user_profile.preferred_end_time,
            "preferred_meal_times": user_profile.preferred_meal_times,
            "feature_vector": getattr(user_profile, "feature_vector", {}),
        }
        logger.info(
            f"[itinerary_optimization_usecase] user_profile_dict: {user_profile_dict}"
        )

        # 통계 정보
        stats_dict = {
            "total_spots": sum(stat["num_spots"] for stat in daily_stats),
            "total_distance_km": sum(stat["total_distance_km"] for stat in daily_stats),
            "total_duration_hours": sum(
                stat["total_duration_hours"] for stat in daily_stats
            ),
        }
        logger.info(f"[itinerary_optimization_usecase.execute] step10 finished")
        logger.info(
            f"[itinerary_optimization_usecase] raw_daily_routes: {daily_routes}"
        )

        stats_dict["daily_stats"] = daily_stats
        logger.info(f"stats_dict with daily_stats: {stats_dict}")

        return ItineraryOptimizationResponse(
            itinerary=itinerary_dict,
            user_profile=user_profile_dict,
            stats=stats_dict,
            daily_routes=daily_stats,
        )


def generate_dummy_users(num_users: int) -> List[UserProfile]:
    """
    num_users만큼의 UserProfile을 생성한다.
    """
    dummy_users = []
    travel_types = ["family", "solo", "couple", "friends", "business"]
    all_themes = [
        "nature",
        "food",
        "culture",
        "shopping",
        "adventure",
        "relaxation",
        "history",
    ]

    for user_id in range(1, num_users + 1):
        travel_type = random.choice(travel_types)
        themes = random.sample(all_themes, k=random.randint(1, 3))

        must_visit_list = list()
        not_visit_list = list()
        user_profile = UserProfile(
            id=user_id,
            travel_type=travel_type,
            group_size=random.randint(1, 5),
            budget_amount=random.randint(100000, 2000000),
            themes=themes,
            preferred_transport="car",
            must_visit_list=must_visit_list,
            not_visit_list=not_visit_list,
        )
        dummy_users.append(user_profile)

    return dummy_users


def generate_dummy_ratings(
        users: List[UserProfile], spots: List[TouristSpot]
) -> List[Tuple[int, int, float]]:
    """
    여러 사용자와 여러 관광지에 대한 (user_id, tourist_spot_id, rating)을 생성.
    must_visit_list, not_visit_list를 반영하고,
    나머지는 1.0~9.0 범위로 랜덤 부여
    """
    dummy_ratings = []
    for user in users:
        for spot in spots:
            if spot.tourist_spot_id in user.must_visit_list:
                rating = 9.0
            elif spot.tourist_spot_id in user.not_visit_list:
                rating = 1.0
            else:
                rating = random.uniform(2.0, 8.0)
            dummy_ratings.append((user.id, spot.tourist_spot_id, rating))
    return dummy_ratings
