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

        # 1. 레거시처럼 user_id, city_id, num_days 등의 인터페이스 유지
        user_profile = request.user_profile
        logger.info(f"[itinerary_optimization_usecase.execute] step1 finished")

        # 2. 관광지 목록 조회
        spots = self.tourist_spot_repository.find_by_city_id(request.city_id)
        if not spots:
            raise ValueError(f"도시에 관광지가 없습니다: {request.city_id}")
        logger.info(f"[itinerary_optimization_usecase.execute] step2 finished")
        # 3. 아직 학습되지 않은 경우, 더미 평점 데이터를 이용해 학습
        if not hasattr(self.user_preference_model.user_features_scaler, "mean_"):
            dummy_users = generate_dummy_users(num_users=100)
            dummy_ratings = generate_dummy_ratings(dummy_users, spots)
            self.user_preference_model.train(dummy_users, spots, dummy_ratings)
        logger.info(f"[itinerary_optimization_usecase.execute] step3 finished")
        # 4. 사용자 선호도 점수 계산
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
        # retain clusters for response building

        # 7. 일자별 경로 최적화
        daily_routes = []
        daily_stats = []

        for day_idx, cluster_spots in enumerate(selected_clusters):
            logger.info(
                f"[step7] day {day_idx+1}: cluster_spots count={len(cluster_spots)}, selected_clusters total={len(selected_clusters)}"
            )
            # 첫째 날은 첫 번째 관광지부터 시작
            # 이후 날짜는 이전 날의 마지막 관광지에서 시작
            if day_idx == 0 or not daily_routes:
                start_spot_idx = 0
            else:
                prev_day_route = daily_routes[-1][0]
                if prev_day_route and len(prev_day_route) >= 2:
                    start_spot_idx = prev_day_route[-2]
                else:
                    start_spot_idx = 0
            # Sanitize start_spot_idx to prevent out-of-range access
            if cluster_spots:
                safe_start = min(max(start_spot_idx, 0), len(cluster_spots) - 1)
                if safe_start != start_spot_idx:
                    logger.warning(
                        f"[step7] day {day_idx+1}: adjusted start_spot_idx from {start_spot_idx} to {safe_start}"
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
                f"[step7] day {day_idx+1}: optimized route={route}, dist={total_dist}, dur={total_dur}"
            )

            logger.info(
                f"[step7] day {day_idx+1}: pre-fallback check, route length={len(route)}"
            )
            if any(idx < 0 or idx >= len(cluster_spots) for idx in route):
                logger.warning(
                    f"[itinerary_optimization_usecase] Invalid route indices for day {day_idx+1}: {route}"
                )
                # Treat as fallback on invalid indices
                if cluster_spots:
                    safe_start = min(max(start_spot_idx, 0), len(cluster_spots) - 1)
                    max_places = request.max_spots_per_day
                    # Select top-scoring spots within this cluster for fallback
                    ordered_indices = sorted(
                        range(len(cluster_spots)),
                        key=lambda i: priority_scores.get(
                            cluster_spots[i].tourist_spot_id, 0.0
                        ),
                        reverse=True,
                    )
                    limited_indices = ordered_indices[:max_places]
                    # For fallback, start at safe_start then visit top spots in cluster (no return loop)
                    if limited_indices and limited_indices[0] == safe_start:
                        fallback_route = limited_indices
                    else:
                        fallback_route = [safe_start] + limited_indices
                else:
                    fallback_route = []
                logger.warning(
                    f"[step7] day {day_idx+1}: invalid indices, fallback_route={fallback_route}"
                )
                # Generate timetable for fallback to compute actual durations and distances
                timetable = self.time_calculation_service.generate_timetable(
                    spots=cluster_spots,
                    route=fallback_route,
                )
                # Compute total travel duration and approximate distance
                fallback_dur = sum(
                    item["duration"] for item in timetable if item["type"] == "travel"
                )
                fallback_dist = fallback_dur * request.transport_speed_kmh
                daily_routes.append((fallback_route, fallback_dist, fallback_dur))
                daily_stats.append(
                    {
                        "day": day_idx + 1,
                        "num_spots": len(limited_indices),
                        "total_distance_km": fallback_dist,
                        "total_duration_hours": fallback_dur,
                        "spots": [
                            cluster_spots[i].tourist_spot_id for i in limited_indices
                        ],
                    }
                )
                continue
            # TODO: Failback 고도화 필요.
            if not route or len(route) < 2:
                # 빈 경로일 때는 최소한 시작지→시작지 순환 경로로 대체하거나,
                # 통계를 빈 값으로 기록하고 다음으로 건너뜁니다.
                if cluster_spots:
                    safe_start = min(max(start_spot_idx, 0), len(cluster_spots) - 1)
                    max_places = request.max_spots_per_day
                    # Select top-scoring spots within this cluster for fallback
                    ordered_indices = sorted(
                        range(len(cluster_spots)),
                        key=lambda i: priority_scores.get(
                            cluster_spots[i].tourist_spot_id, 0.0
                        ),
                        reverse=True,
                    )
                    limited_indices = ordered_indices[:max_places]
                    # For fallback, start at safe_start then visit top spots in cluster (no return loop)
                    if limited_indices and limited_indices[0] == safe_start:
                        fallback_route = limited_indices
                    else:
                        fallback_route = [safe_start] + limited_indices
                else:
                    fallback_route = []
                logger.warning(
                    f"[step7] day {day_idx+1}: empty or too short route, fallback_route={fallback_route}"
                )
                # Generate timetable for fallback to compute actual durations and distances
                timetable = self.time_calculation_service.generate_timetable(
                    spots=cluster_spots,
                    route=fallback_route,
                )
                # Compute total travel duration and approximate distance
                fallback_dur = sum(
                    item["duration"] for item in timetable if item["type"] == "travel"
                )
                fallback_dist = fallback_dur * request.transport_speed_kmh
                daily_routes.append((fallback_route, fallback_dist, fallback_dur))
                daily_stats.append(
                    {
                        "day": day_idx + 1,
                        "num_spots": len(limited_indices),
                        "total_distance_km": fallback_dist,
                        "total_duration_hours": fallback_dur,
                        "spots": [
                            cluster_spots[i].tourist_spot_id for i in limited_indices
                        ],
                    }
                )
                # 연속성 계산용 prev_day_end는 업데이트하지 않음
                continue

            daily_routes.append((route, total_dist, total_dur))
            logger.info(f"[step7] day {day_idx+1}: daily_routes now={daily_routes}")

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
        # Log collected daily_stats after loop and before step8
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
                f"[step8] day {day_idx+1}: preparing timetable, day_spots count={len(selected_clusters[day_idx])}, route={route}"
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
        # 9. DB 저장 로직 생략 -> 생성된 itinerary를 그대로 사용
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

        # Log itinerary_dict after definition, before populating days
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
        # Log user_profile_dict after definition
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

        # Reuse daily_stats directly for response
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
        # 랜덤하게 travel_type, theme 등을 선정
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
