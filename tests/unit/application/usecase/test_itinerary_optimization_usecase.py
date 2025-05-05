from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from application.usecase.itinerary_optimization_usecase import (
    ItineraryOptimizationRequest, ItineraryOptimizationResponse,
    ItineraryOptimizationUseCase)
from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.value_objects.coordinate import Coordinate


@pytest.fixture
def mock_dependencies():
    tourist_spot_repository = MagicMock()
    clustering_service = MagicMock()
    route_optimization_service = MagicMock()
    score_calculation_service = MagicMock()
    time_calculation_service = MagicMock()
    user_preference_model = MagicMock()
    enhanced_clustering_algorithm = MagicMock()

    return {
        "tourist_spot_repository": tourist_spot_repository,
        "clustering_service": clustering_service,
        "route_optimization_service": route_optimization_service,
        "score_calculation_service": score_calculation_service,
        "time_calculation_service": time_calculation_service,
        "user_preference_model": user_preference_model,
        "enhanced_clustering_algorithm": enhanced_clustering_algorithm,
    }


@pytest.fixture
def sample_user_profile():
    return UserProfile(
        travel_type="family",
        group_size=4,
        budget_amount=500000,
        themes=["nature", "food"],
        must_visit_list=[101, 102],
        not_visit_list=[999],
        preferred_transport="car",
        preferred_activity_level=7.0,
        preferred_start_time="09:00",
        preferred_end_time="20:00",
        preferred_meal_times={
            "breakfast": "08:30",
            "lunch": "12:30",
            "dinner": "19:00",
        },
    )


@pytest.fixture
def sample_request(sample_user_profile):
    return ItineraryOptimizationRequest(
        user_id=12345,
        city_id=1,
        num_days=3,
        user_profile=sample_user_profile,
        start_date=date(2025, 5, 1),
        max_spots_per_day=5,
        max_distance_per_day_km=100.0,
        max_duration_per_day_hours=8.0,
        transport_speed_kmh=40.0,
        optimize_for_theme=True,
        optimize_for_distance=True,
        optimize_for_time=True,
        optimize_for_popularity=True,
        optimize_for_user_preference=True,
        optimize_for_day_connectivity=True,
    )


@pytest.fixture
def sample_spots():
    spot1 = TouristSpot(
        tourist_spot_id="101",
        name="Spot 101",
        coordinate=Coordinate(latitude=37.123, longitude=127.123),
        category=["자연", "명소"],
        business_hours="매일 09:00-18:00",
        opening_hours="09:00 - 18:00",
        average_visit_duration=2.0,
    )
    spot2 = TouristSpot(
        tourist_spot_id=102,
        name="Spot 102",
        coordinate=Coordinate(latitude=37.124, longitude=127.125),
        category=["체험"],
        activity_level="3.5",
        business_status="영업 중",
        opening_hours="08:00 - 20:00",
    )
    spot3 = TouristSpot(
        tourist_spot_id=103,
        name="Spot 103",
        coordinate=Coordinate(latitude=37.200, longitude=127.300),
        category="역사",
        opening_hours="10:00 - 17:00",
    )
    spot4 = TouristSpot(
        tourist_spot_id="999",
        name="Spot 999",
        coordinate=Coordinate(latitude=38.000, longitude=128.000),
        opening_hours="10:00 - 20:00",
    )

    return [spot1, spot2, spot3, spot4]


def test_execute_success(mock_dependencies, sample_request, sample_spots):
    """
    정상 시나리오 테스트:
    """

    mock_dependencies["tourist_spot_repository"].find_by_city_id.return_value = (
        sample_spots
    )
    mock_dependencies["user_preference_model"].user_features_scaler = MagicMock()
    mock_dependencies["user_preference_model"].user_features_scaler.mean_ = 0.0
    mock_dependencies["user_preference_model"].predict_preference.return_value = 0.5
    mock_dependencies["score_calculation_service"].calculate_base_score.return_value = (
        1.0
    )
    mock_dependencies[
        "score_calculation_service"
    ].calculate_popularity_score.return_value = 2.0
    mock_dependencies[
        "score_calculation_service"
    ].calculate_theme_match_score.return_value = 3.0
    mock_dependencies[
        "enhanced_clustering_algorithm"
    ].cluster_spots_for_multi_day_trip.return_value = [
        [sample_spots[0], sample_spots[1]],
        [sample_spots[2]],
        [sample_spots[3]],
    ]
    mock_dependencies[
        "enhanced_clustering_algorithm"
    ].select_optimal_clusters.return_value = [
        [sample_spots[0], sample_spots[1]],
        [sample_spots[2]],
        [sample_spots[3]],
    ]
    mock_dependencies["route_optimization_service"].optimize_route.side_effect = [
        (
            [0, 1, 0],
            10.0,
            2.0,
        ),
        ([0, 0], 12.0, 3.0),
        ([0, 0], 15.0, 4.0),
    ]
    mock_dependencies["time_calculation_service"].generate_timetable.side_effect = [
        [
            {
                "type": "visit",
                "spot_id": sample_spots[0].tourist_spot_id,
                "spot_name": sample_spots[0].name,
                "start_time": datetime(2025, 5, 1, 9, 0),
                "end_time": datetime(2025, 5, 1, 10, 0),
                "duration": 1.0,
            },
            {
                "type": "travel",
                "from_spot_id": sample_spots[0].tourist_spot_id,
                "from_spot_name": sample_spots[0].name,
                "to_spot_id": sample_spots[1].tourist_spot_id,
                "to_spot_name": sample_spots[1].name,
                "start_time": datetime(2025, 5, 1, 10, 0),
                "end_time": datetime(2025, 5, 1, 10, 30),
                "duration": 0.5,
            },
            {
                "type": "visit",
                "spot_id": sample_spots[1].tourist_spot_id,
                "spot_name": sample_spots[1].name,
                "start_time": datetime(2025, 5, 1, 10, 30),
                "end_time": datetime(2025, 5, 1, 12, 0),
                "duration": 1.5,
            },
        ],
        [
            {
                "type": "visit",
                "spot_id": sample_spots[2].tourist_spot_id,
                "spot_name": sample_spots[2].name,
                "start_time": datetime(2025, 5, 2, 9, 0),
                "end_time": datetime(2025, 5, 2, 11, 0),
                "duration": 2.0,
            }
        ],
        [
            {
                "type": "visit",
                "spot_id": sample_spots[3].tourist_spot_id,
                "spot_name": sample_spots[3].name,
                "start_time": datetime(2025, 5, 3, 10, 0),
                "end_time": datetime(2025, 5, 3, 12, 0),
                "duration": 2.0,
            }
        ],
    ]

    use_case = ItineraryOptimizationUseCase(**mock_dependencies)
    response: ItineraryOptimizationResponse = use_case.execute(sample_request)
    assert isinstance(response, ItineraryOptimizationResponse)
    assert isinstance(response.itinerary, dict)
    assert "days" in response.itinerary
    assert isinstance(response.user_profile, dict)
    assert isinstance(response.stats, dict)
    assert isinstance(response.daily_routes, list)
    mock_dependencies[
        "tourist_spot_repository"
    ].find_by_city_id.assert_called_once_with(sample_request.city_id)
    mock_dependencies[
        "enhanced_clustering_algorithm"
    ].cluster_spots_for_multi_day_trip.assert_called_once()
    mock_dependencies[
        "enhanced_clustering_algorithm"
    ].select_optimal_clusters.assert_called_once()
    assert (
        mock_dependencies["route_optimization_service"].optimize_route.call_count == 3
    )
    assert (
        mock_dependencies["time_calculation_service"].generate_timetable.call_count == 3
    )
    itinerary_days = response.itinerary["days"]
    assert len(itinerary_days) == 3
    first_day = itinerary_days[0]
    assert first_day["day_number"] == 1
    assert len(first_day["items"]) == 3
    assert first_day["items"][0]["type"] == "visit"
    assert first_day["items"][1]["type"] == "travel"
    assert first_day["items"][2]["type"] == "visit"
    assert len(response.daily_routes) == 3
    assert response.daily_routes[0]["day"] == 1
    assert response.daily_routes[0]["total_distance_km"] == 10.0
    assert response.daily_routes[0]["total_duration_hours"] == 2.0
    total_distance = response.stats["total_distance_km"]
    total_duration = response.stats["total_duration_hours"]
    assert total_distance == pytest.approx(37.0)
    assert total_duration == pytest.approx(9.0)
    up = response.user_profile
    assert up["travel_type"] == "family"
    assert up["group_size"] == 4
    assert 101 in up["must_visit_list"]
    assert 999 in up["not_visit_list"]
