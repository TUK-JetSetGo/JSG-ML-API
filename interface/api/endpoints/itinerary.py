"""
여행 일정 최적화 API 엔드포인트 모듈 (itinerary_repository 없이)
"""

import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from application.usecase.itinerary_optimization_usecase import (
    ItineraryOptimizationRequest, ItineraryOptimizationResponse,
    ItineraryOptimizationUseCase)
from domain.services.clustering_service import ClusteringService
from domain.services.route_optimization_service import RouteOptimizationService
from domain.services.score_calculation_service import ScoreCalculationService
from domain.services.time_calculation_service import TimeCalculationService
from infrastructure.adapters.ml.enhanced_clustering_algorithm import \
    EnhancedClusteringAlgorithm
from infrastructure.adapters.ml.user_preference_model import \
    UserPreferenceModel
from infrastructure.adapters.repositories.tourist_spot_repository_impl import \
    TouristSpotRepositoryImpl


# 3) 사용자 프로필 입력 모델
class UserProfileInputModel(BaseModel):
    travel_type: str
    group_size: int
    budget_amount: int
    themes: List[str] = []
    must_visit_list: List[int] = []
    not_visit_list: List[int] = []
    preferred_transport: str = "car"
    preferred_activity_level: Optional[float] = None
    preferred_start_time: Optional[str] = None
    preferred_end_time: Optional[str] = None
    preferred_meal_times: Dict[str, str] = {}


# 4) 요청 모델 (itinerary_repository 전혀 안 씀)
class ItineraryOptimizationRequestModel(BaseModel):
    user_id: int
    city_id: int
    num_days: int
    user_profile: UserProfileInputModel

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


# 5) 응답 모델
class ItineraryDayItemModel(BaseModel):
    item_number: int
    type: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: float
    spot_id: Optional[int] = None
    spot_name: Optional[str] = None
    from_spot_id: Optional[int] = None
    from_spot_name: Optional[str] = None
    to_spot_id: Optional[int] = None
    to_spot_name: Optional[str] = None


class ItineraryDayModel(BaseModel):
    day_number: int
    items: List[ItineraryDayItemModel]


class ItineraryModel(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: List[ItineraryDayModel]


class UserProfileModel(BaseModel):
    travel_type: str
    group_size: int
    budget_amount: int
    themes: List[str]
    must_visit_list: List[int]
    not_visit_list: List[int]
    preferred_transport: str
    preferred_activity_level: Optional[float]
    preferred_start_time: Optional[str]
    preferred_end_time: Optional[str]
    preferred_meal_times: Dict[str, str]
    feature_vector: Dict[str, float] = {}


class DailyStatModel(BaseModel):
    day: int
    num_spots: int
    total_distance_km: float
    total_duration_hours: float
    spots: List[int]


class StatsModel(BaseModel):
    total_spots: int
    total_distance_km: float
    total_duration_hours: float
    daily_stats: List[DailyStatModel]


class ItineraryOptimizationResponseModel(BaseModel):
    itinerary: ItineraryModel
    user_profile: UserProfileModel
    stats: StatsModel
    daily_routes: List[DailyStatModel]


# 6) 의존성 주입 - itinerary_repository 제거
def get_itinerary_optimization_usecase() -> ItineraryOptimizationUseCase:
    tourist_spot_repository = TouristSpotRepositoryImpl()
    clustering_service = ClusteringService()
    route_optimization_service = RouteOptimizationService()
    score_calculation_service = ScoreCalculationService()
    time_calculation_service = TimeCalculationService()
    user_preference_model = UserPreferenceModel()
    enhanced_clustering_algorithm = EnhancedClusteringAlgorithm()

    return ItineraryOptimizationUseCase(
        tourist_spot_repository=tourist_spot_repository,
        clustering_service=clustering_service,
        route_optimization_service=route_optimization_service,
        score_calculation_service=score_calculation_service,
        time_calculation_service=time_calculation_service,
        user_preference_model=user_preference_model,
        enhanced_clustering_algorithm=enhanced_clustering_algorithm,
    )


# 7) 라우터 정의 (오직 /optimize 만)
router = APIRouter(prefix="/api/v1/itineraries", tags=["itineraries"])


@router.post(
    "/optimize",
    response_model=ItineraryOptimizationResponseModel,
    summary="여행 일정 최적화",
    description="관광지 목록만 DB에서 불러오고, 사용자 프로필은 직접 입력받아서 일정 최적화 (itinerary_repository 없음).",
)
async def optimize_itinerary(
    request_data: ItineraryOptimizationRequestModel,
    usecase: ItineraryOptimizationUseCase = Depends(get_itinerary_optimization_usecase),
) -> ItineraryOptimizationResponseModel:
    """
    itinerary_repository 없이 동작하는 일정 최적화 엔드포인트
    """
    try:
        # 1) user_profile 도메인 객체 생성
        from domain.entities.user_profile import UserProfile

        user_profile = UserProfile(
            travel_type=request_data.user_profile.travel_type,
            group_size=request_data.user_profile.group_size,
            budget_amount=request_data.user_profile.budget_amount,
            themes=request_data.user_profile.themes,
            must_visit_list=request_data.user_profile.must_visit_list,
            not_visit_list=request_data.user_profile.not_visit_list,
            preferred_transport=request_data.user_profile.preferred_transport,
            preferred_activity_level=request_data.user_profile.preferred_activity_level,
            preferred_start_time=request_data.user_profile.preferred_start_time,
            preferred_end_time=request_data.user_profile.preferred_end_time,
            preferred_meal_times=request_data.user_profile.preferred_meal_times,
        )

        # 2) 유즈케이스 요청 객체
        new_request = ItineraryOptimizationRequest(
            user_id=request_data.user_id,
            city_id=request_data.city_id,
            num_days=request_data.num_days,
            user_profile=user_profile,
            start_date=request_data.start_date,
            max_spots_per_day=request_data.max_spots_per_day,
            max_distance_per_day_km=request_data.max_distance_per_day_km,
            max_duration_per_day_hours=request_data.max_duration_per_day_hours,
            transport_speed_kmh=request_data.transport_speed_kmh,
            optimize_for_theme=request_data.optimize_for_theme,
            optimize_for_distance=request_data.optimize_for_distance,
            optimize_for_time=request_data.optimize_for_time,
            optimize_for_popularity=request_data.optimize_for_popularity,
            optimize_for_user_preference=request_data.optimize_for_user_preference,
            optimize_for_day_connectivity=request_data.optimize_for_day_connectivity,
        )

        response: ItineraryOptimizationResponse = usecase.execute(new_request)
        return ItineraryOptimizationResponseModel(**response.__dict__)

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(ex)}")
