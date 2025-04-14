import math
from datetime import datetime

from domain.services.score_calculation_service import ScoreCalculationService


class DummyCoordinate:
    """테스트용 좌표 클래스 (최소한의 distance_to 구현)."""

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
        # 여기서는 단순 계산을 위해 Euclidean 거리를 사용합니다.
        return (
            (self.latitude - other.latitude) ** 2
            + (self.longitude - other.longitude) ** 2
        ) ** 0.5


class DummyTouristSpot:
    """
    테스트용 TouristSpot 클래스.
    - activity_level: float 값
    - categories: 문자열 목록
    - opening_time, closing_time: datetime 객체
    """

    def __init__(
        self,
        spot_id: int,
        activity_level: float,
        latitude: float,
        longitude: float,
        categories=None,
        opening_time: datetime = None,
        closing_time: datetime = None,
        average_visit_duration: float = 1.0,
    ):
        self.id = spot_id
        self.activity_level = activity_level
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.categories = categories if categories is not None else []
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.average_visit_duration = average_visit_duration


class DummyUserProfile:
    """
    Dummy implementation of UserProfile for testing.
    themes: 사용자 테마 (카테고리 매칭 점수 계산에 사용)
    must_visit_list: 반드시 방문해야 하는 관광지의 id 리스트
    not_visit_list: 방문하지 않아야 하는 관광지의 id 리스트
    """

    def __init__(self, themes=None, must_visit_list=None, not_visit_list=None):
        self.themes = themes if themes is not None else []
        self.must_visit_list = must_visit_list if must_visit_list is not None else []
        self.not_visit_list = not_visit_list if not_visit_list is not None else []


def test_calculate_base_score_without_user_profile():
    """
    user_profile이 없을 경우, 기본 점수가 activity_level * base_scale로 반환되는지 테스트합니다.
    """
    spot = DummyTouristSpot(spot_id=1, activity_level=5.0, latitude=0, longitude=0)
    # user_profile이 None인 경우, 카테고리 점수나 추가 가중치가 적용되지 않아야 함.
    base_score = ScoreCalculationService.calculate_base_score(
        spot, None, base_scale=1.0
    )
    assert math.isclose(base_score, 5.0, rel_tol=1e-5)


def test_calculate_base_score_with_user_profile_no_extra():
    """
    user_profile이 제공되었지만, must_visit_list와 not_visit_list에 해당 spot.id가 없고,
    카테고리 관련 항목도 비어 있다면, 기본 점수에 카테고리 매칭 점수 (0.0)가 더해져 동일한 값이어야 함.
    """
    spot = DummyTouristSpot(
        spot_id=1, activity_level=5.0, latitude=0, longitude=0, categories=[]
    )
    profile = DummyUserProfile(themes=[], must_visit_list=[], not_visit_list=[])
    base_score = ScoreCalculationService.calculate_base_score(
        spot, profile, base_scale=1.0
    )
    # 카테고리 매칭 점수가 0.0 이므로 결과는 5.0이어야 함.
    assert math.isclose(base_score, 5.0, rel_tol=1e-5)


def test_calculate_base_score_must_visit():
    """
    user_profile의 must_visit_list에 spot.id가 포함된 경우,
    기본 점수에 1000.0이 추가되어 반환되는지 테스트합니다.
    """
    spot = DummyTouristSpot(
        spot_id=1, activity_level=5.0, latitude=0, longitude=0, categories=[]
    )
    # must_visit_list에 1 포함, not_visit_list는 비어있음.
    profile = DummyUserProfile(themes=[], must_visit_list=[1], not_visit_list=[])
    base_score = ScoreCalculationService.calculate_base_score(
        spot, profile, base_scale=1.0
    )
    expected = 5.0 + 1000.0
    assert math.isclose(base_score, expected, rel_tol=1e-5)


def test_calculate_base_score_not_visit():
    """
    user_profile의 not_visit_list에 spot.id가 포함된 경우,
    기본 점수가 무조건 -1000.0으로 설정되는지 테스트합니다.
    """
    spot = DummyTouristSpot(
        spot_id=1, activity_level=5.0, latitude=0, longitude=0, categories=[]
    )
    # not_visit_list에 1 포함. must_visit_list는 비어있음.
    profile = DummyUserProfile(themes=[], must_visit_list=[], not_visit_list=[1])
    base_score = ScoreCalculationService.calculate_base_score(
        spot, profile, base_scale=1.0
    )
    expected = -1000.0
    assert math.isclose(base_score, expected, rel_tol=1e-5)


def test_calculate_base_score_with_both_lists():
    """
    만약 동일한 spot.id가 must_visit_list와 not_visit_list 모두에 포함되면,
    최종 점수는 not_visit_list의 처리가 우선하여 -1000.0이 되어야 합니다.
    """
    spot = DummyTouristSpot(
        spot_id=1, activity_level=5.0, latitude=0, longitude=0, categories=[]
    )
    profile = DummyUserProfile(themes=[], must_visit_list=[1], not_visit_list=[1])
    base_score = ScoreCalculationService.calculate_base_score(
        spot, profile, base_scale=1.0
    )
    expected = -1000.0  # not_visit_list가 우선함.
    assert math.isclose(base_score, expected, rel_tol=1e-5)
