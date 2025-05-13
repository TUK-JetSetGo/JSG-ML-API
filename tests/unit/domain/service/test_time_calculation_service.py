import datetime
import math

from domain.services.time_calculation_service import TimeCalculationService


class DummyCoordinate:
    """테스트용 좌표 클래스 (Euclidean 거리를 사용)."""

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
        # 단순 Euclidean 거리 계산.
        return (
            (self.latitude - other.latitude) ** 2
            + (self.longitude - other.longitude) ** 2
        ) ** 0.5


class DummyTouristSpot:
    """
    테스트용 TouristSpot 클래스.

    필요한 속성:
      - tourist_spot_id: int
      - name: str
      - coordinate: DummyCoordinate (latitude, longitude)
      - average_visit_duration: float (방문 소요 시간, 시간 단위)
      - opening_time, closing_time: datetime.time
    """

    def __init__(
        self,
        tourist_spot_id: int,
        name: str,
        activity_level: float,
        latitude: float,
        longitude: float,
        average_visit_duration: float,
        opening_time: datetime.time = datetime.time(9, 0),
        closing_time: datetime.time = datetime.time(17, 0),
    ):
        self.tourist_spot_id = tourist_spot_id
        self.name = name
        self.activity_level = activity_level
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.average_visit_duration = average_visit_duration
        self.opening_time = opening_time
        self.closing_time = closing_time


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # 지구 반경 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def test_calculate_travel_time():
    """
    두 관광지 간 이동 시간을 올바르게 계산하는지 테스트합니다.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spot_from = DummyTouristSpot(
        tourist_spot_id=1,
        name="A",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.0,
        average_visit_duration=1.0,
    )
    spot_to = DummyTouristSpot(
        tourist_spot_id=2,
        name="B",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.359,
        average_visit_duration=1.0,
    )
    travel_time = service.calculate_travel_time(spot_from, spot_to)

    assert math.isclose(travel_time, 1.0, rel_tol=1e-3)


def test_calculate_route_times():
    """
    여러 관광지를 방문하는 경로에서 총 이동 시간과 방문 시간, 세부 시간을 올바르게 계산하는지 테스트합니다.
    생산 코드의 Coordinate.distance_to (Haversine 공식 사용)에 따라 계산된 이동 시간을 기준으로 기대값을 설정합니다.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spot0 = DummyTouristSpot(1, "A", 5.0, 0.0, 0.0, average_visit_duration=1.0)
    spot1 = DummyTouristSpot(2, "B", 5.0, 3.0, 4.0, average_visit_duration=2.0)
    spot2 = DummyTouristSpot(3, "C", 5.0, 6.0, 8.0, average_visit_duration=1.5)
    spots = [spot0, spot1, spot2]
    route = [0, 1, 2, 0]

    travel_time, visit_time, detailed_times = service.calculate_route_times(
        spots, route
    )

    # 각 구간별 이동 거리를 Haversine 공식으로 계산.
    leg1_distance = haversine(0.0, 0.0, 3.0, 4.0)
    leg2_distance = haversine(3.0, 4.0, 6.0, 8.0)
    leg3_distance = haversine(6.0, 8.0, 0.0, 0.0)
    expected_leg1 = leg1_distance / 40.0
    expected_leg2 = leg2_distance / 40.0
    expected_leg3 = leg3_distance / 40.0
    expected_travel = expected_leg1 + expected_leg2 + expected_leg3
    expected_visit = 2.0 + 1.5

    # 확인: 실제 계산된 이동 시간과 예상값이 일치하는지 (허용 오차 1e-2)
    assert math.isclose(
        travel_time, expected_travel, rel_tol=1e-2
    ), f"Calculated travel_time: {travel_time}, expected: {expected_travel}"
    assert math.isclose(
        visit_time, expected_visit, rel_tol=1e-2
    ), f"Calculated visit_time: {visit_time}, expected: {expected_visit}"
    # 상세 정보 길이는 leg 수와 동일해야 함.
    assert len(detailed_times) == 3


def test_generate_timetable_length2():
    """
    경로 길이가 2인 경우(출발지와 도착지 간),
    일정표가 올바르게 생성되는지 검증합니다.

    예상:
      - 첫 항목: 출발 관광지 방문 (방문 시간 = average_visit_duration)
      - 두 번째 항목: 출발 → 도착 관광지 이동 (이동 시간 계산)
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(
        tourist_spot_id=1,
        name="A",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.0,
        average_visit_duration=1.0,
    )
    # 0.359도의 차이는 약 40 km
    spotB = DummyTouristSpot(
        tourist_spot_id=2,
        name="B",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.359,
        average_visit_duration=1.5,
    )
    spots = [spotA, spotB]
    route = [0, 1]
    start_time = datetime.time(9, 0)

    timetable = service.generate_timetable(spots, route, start_time)
    # 예상 항목 수는 2
    assert len(timetable) == 2
    visit_entry = timetable[0]
    travel_entry = timetable[1]
    assert visit_entry["type"] == "visit"
    assert visit_entry["spot_id"] == spotA.tourist_spot_id
    assert visit_entry["spot_name"] == spotA.name
    # 방문: 시작 09:00, 방문 시간 1.0h → 종료 약 10:00
    assert visit_entry["start_time"].hour == 9
    # 종료 시간이 10시 또는 10시 근처인지 확인 (분, 초는 상관없음)
    # 여기서는 10시가 정확히 나와야 함.
    assert visit_entry["end_time"].hour == 10

    assert travel_entry["type"] == "travel"
    assert travel_entry["from_spot_id"] == spotA.tourist_spot_id
    assert travel_entry["to_spot_id"] == spotB.tourist_spot_id
    # 이동: 40km/40 km/h = 1.0h, 10:00 시작 → 기대: 11:00 종료
    assert travel_entry["start_time"].hour == 10

    # travel_entry['end_time']가 10시 59분 정도로 계산될 수 있으므로, 종료 시간이 11시와 1분 미만 차이 나는지 확인합니다.
    def time_to_seconds(t: datetime.time) -> int:
        return t.hour * 3600 + t.minute * 60 + t.second

    expected_end = datetime.time(11, 0)
    diff = abs(
        time_to_seconds(travel_entry["end_time"]) - time_to_seconds(expected_end)
    )
    assert diff < 60, f"Time difference too high: {diff} seconds (expected near 11:00)"


def test_generate_timetable_length3():
    """
    경로 길이가 3인 경우 (예: [0, 1, 0]) 일정표 생성 검증.
    예상:
      - 첫 구간(i=0): no visit, travel entry from spot0 -> spot1.
      - 두 번째 구간(i=1): visit entry at spot1.
    총 예상 항목 수: 2.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(1, "A", 5.0, 0.0, 0.0, 1.0)
    # 0.359도 차이는 약 40 km (적도 기준)
    spotB = DummyTouristSpot(2, "B", 5.0, 0.0, 0.359, 1.5)
    spots = [spotA, spotB]
    route = [0, 1, 0]
    start_time = datetime.time(9, 0)
    timetable = service.generate_timetable(spots, route, start_time)
    assert (
        len(timetable) == 2
    ), f"Expected 2 entries for route length 3, got {len(timetable)}"


def test_generate_timetable_length4():
    """
    경로 길이가 4인 경우 (예: [0, 1, 2, 0]) 일정표 생성 검증.
    예상:
      - detailed_times 길이: 3.
      - i=0: no visit, travel entry from spot0 -> spot1.
      - i=1: visit entry at spot1, travel entry from spot1 -> spot2.
      - i=2: visit entry at spot2, no travel entry.
    총 예상 항목 수: 4.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(1, "A", 5.0, 0.0, 0.0, 1.0)
    spotB = DummyTouristSpot(2, "B", 5.0, 0.0, 0.359, 2.0)
    spotC = DummyTouristSpot(3, "C", 5.0, 0.0, 0.718, 1.5)
    spots = [spotA, spotB, spotC]
    route = [0, 1, 2, 0]
    start_time = datetime.time(9, 0)
    timetable = service.generate_timetable(spots, route, start_time)
    assert (
        len(timetable) == 4
    ), f"Expected 4 entries for route length 4, got {len(timetable)}"


def test_generate_timetable_length5():
    """
    경로 길이가 5인 경우 (예: [0, 1, 2, 3, 0]) 일정표 생성 검증.
    예상:
      - detailed_times 길이: 4.
      - i=0: no visit, travel entry from spotA -> spotB.
      - i=1: visit entry at spotB, travel entry from spotB -> spotC.
      - i=2: visit entry at spotC, travel entry from spotC -> spotD.
      - i=3: visit entry at spotD, no travel entry.
    총 예상 항목 수: 6.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(1, "A", 5.0, 0.0, 0.0, 1.0)
    spotB = DummyTouristSpot(2, "B", 5.0, 0.0, 0.359, 1.5)
    spotC = DummyTouristSpot(3, "C", 5.0, 0.0, 0.718, 2.0)
    spotD = DummyTouristSpot(4, "D", 5.0, 0.0, 1.077, 2.5)
    spots = [spotA, spotB, spotC, spotD]
    route = [0, 1, 2, 3, 0]
    start_time = datetime.time(9, 0)
    timetable = service.generate_timetable(spots, route, start_time)
    assert (
        len(timetable) == 6
    ), f"Expected 6 entries for route length 5, got {len(timetable)}"


def test_check_time_constraints_within_limit():
    """
    총 소요 시간이 max_duration 이내인 경우 True와 일정표를 반환하는지 검증합니다.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(
        tourist_spot_id=1,
        name="A",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.0,
        average_visit_duration=1.0,
    )
    # 0.359도 차이는 약 40 km → travel time 약 1.0h
    spotB = DummyTouristSpot(
        tourist_spot_id=2,
        name="B",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.359,
        average_visit_duration=1.0,
    )
    spots = [spotA, spotB]
    route = [0, 1]
    within, timetable = service.check_time_constraints(
        spots, route, max_duration=3.0, start_time=datetime.time(9, 0)
    )
    assert within is True
    assert timetable is not None


def test_check_time_constraints_exceed_limit():
    """
    총 소요 시간이 max_duration 을 초과하면 False와 None을 반환하는지 검증합니다.
    경로의 경우, [0, 1]에서는 첫 구간에 방문 시간이 포함되지 않고 이동 시간만 계산되므로,
    max_duration을 이동 시간보다 낮은 값(예: 0.5시간)으로 설정하여 제약 조건 초과 상황을 만듭니다.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(
        tourist_spot_id=1,
        name="A",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.0,
        average_visit_duration=2.0,
    )
    # 0.359도의 차이는 약 40 km → 이동 시간은 약 1.0 시간 정도
    spotB = DummyTouristSpot(
        tourist_spot_id=2,
        name="B",
        activity_level=5.0,
        latitude=0.0,
        longitude=0.359,
        average_visit_duration=2.0,
    )
    spots = [spotA, spotB]
    route = [0, 1]
    # max_duration을 0.5시간으로 설정하면 전체 소요 시간(이동 시간 약 1.0h)이 초과되어야 합니다.
    within, timetable = service.check_time_constraints(
        spots, route, max_duration=0.5, start_time=datetime.time(9, 0)
    )
    assert within is False
    assert timetable is None


def test_adjust_route_for_time_constraints():
    """
    시간 제약 조건에 맞게 경로에서 우선순위가 낮은 관광지를 제거하는지 검증합니다.

    예시:
      - 4개 관광지 (A, B, C, D)를 포함하는 경로: A -> B -> C -> D -> A
      - 각 관광지의 방문 시간이 길어 전체 소요 시간이 max_duration을 초과함.
      - priority_scores를 제공하여 우선순위가 높은 관광지만 남도록 조정합니다.

    테스트에서는 우선순위가 가장 높은 관광지 C만 남아 최종 경로가 [A, C, A]가 되어야 함.
    """
    service = TimeCalculationService(transport_speed_kmh=40.0)
    spotA = DummyTouristSpot(1, "A", 5.0, 0.0, 0.0, average_visit_duration=2.0)
    spotB = DummyTouristSpot(2, "B", 5.0, 0.0, 0.1, average_visit_duration=2.0)
    spotC = DummyTouristSpot(3, "C", 5.0, 0.0, 0.2, average_visit_duration=2.0)
    spotD = DummyTouristSpot(4, "D", 5.0, 0.0, 0.3, average_visit_duration=2.0)
    spots = [spotA, spotB, spotC, spotD]
    route = [0, 1, 2, 3, 0]
    # 전체 방문 시간: B, C, D 각각 2.0h = 6.0h; 이동 시간는 소폭
    # max_duration을 5.0 시간으로 설정하면 초과됨.
    # priority_scores: 우선순위가 높은 값일수록 중요 (B:1.0, C:5.0, D:2.0)
    priority_scores = {
        spotB.tourist_spot_id: 1.0,
        spotC.tourist_spot_id: 5.0,
        spotD.tourist_spot_id: 2.0,
    }
    adjusted_route = service.adjust_route_for_time_constraints(
        spots, route, max_duration=5.0, priority_scores=priority_scores
    )
    # 예상: 시작 A와 우선순위가 가장 높은 C만 남아서 최종 경로가 [A, C, A]가 되어야 함.
    expected_route = [0, 2, 0]
    assert adjusted_route == expected_route
