import datetime
from typing import Any, Dict, List, Optional, Tuple

from domain.entities.tourist_spot import TouristSpot


class TimeCalculationService:
    """관광 시간과 이동 시간 계산을 위한 서비스"""

    def __init__(self, transport_speed_kmh: float = 40.0):
        """
        초기화

        Args:
            transport_speed_kmh: 이동 속도 (km/h)
        """
        self.transport_speed_kmh = transport_speed_kmh

    def calculate_travel_time(
        self, from_spot: TouristSpot, to_spot: TouristSpot
    ) -> float:
        """
        두 관광지 간 이동 시간 계산

        Args:
            from_spot: 출발 관광지
            to_spot: 도착 관광지

        Returns:
            이동 시간 (시간)
        """
        from domain.value_objects.coordinate import Coordinate

        from_coord = Coordinate(
            from_spot.coordinate.latitude, from_spot.coordinate.longitude
        )
        to_coord = Coordinate(to_spot.coordinate.latitude, to_spot.coordinate.longitude)

        # 거리 계산 (km)
        distance = from_coord.distance_to(to_coord)

        # 이동 시간 계산 (시간)
        travel_time = distance / self.transport_speed_kmh

        return travel_time

    def calculate_route_times(
        self, spots: List[TouristSpot], route: List[int]
    ) -> Tuple[float, float, List[Tuple[int, float, float]]]:
        """
        경로의 총 이동 시간과 관광 시간 계산

        Args:
            spots: 관광지 목록
            route: 경로 인덱스 목록

        Returns:
            (총 이동 시간, 총 관광 시간, 세부 시간 정보 목록)
        """
        if not route or len(route) < 2:
            return 0.0, 0.0, []

        total_travel_time = 0.0
        total_visit_time = 0.0
        detailed_times = []

        for i in range(len(route) - 1):
            from_idx = route[i]
            to_idx = route[i + 1]

            # 시작 지점으로 돌아가는 경우 방문 시간 제외
            if i > 0:
                visit_time = spots[from_idx].average_visit_duration
                total_visit_time += visit_time
            else:
                visit_time = 0.0

            # 이동 시간 계산
            travel_time = self.calculate_travel_time(spots[from_idx], spots[to_idx])
            total_travel_time += travel_time

            detailed_times.append((from_idx, visit_time, travel_time))

        return total_travel_time, total_visit_time, detailed_times

    def generate_timetable(
        self,
        spots: List[TouristSpot],
        route: List[int],
        start_time: datetime.time = datetime.time(9),
    ) -> List[Dict[str, Any]]:
        if not route or len(route) < 2:
            return []

        # 이동 시간과 관광 시간 계산
        _, _, detailed_times = self.calculate_route_times(spots, route)
        current_time = datetime.datetime.combine(datetime.date.today(), start_time)
        timetable = []

        # 만약 route 길이가 2라면, 별도로 처리
        if len(route) == 2:
            # 방문 항목: 시작 관광지 방문
            visit_start = current_time
            visit_end = visit_start + datetime.timedelta(
                hours=spots[route[0]].average_visit_duration
            )
            timetable.append(
                {
                    "type": "visit",
                    "spot_id": spots[route[0]].tourist_spot_id,
                    "spot_name": spots[route[0]].name,
                    "start_time": visit_start.time(),
                    "end_time": visit_end.time(),
                    "duration": spots[route[0]].average_visit_duration,
                }
            )
            current_time = visit_end
            # 이동 항목: 시작 -> 도착 관광지 이동
            travel_time = self.calculate_travel_time(spots[route[0]], spots[route[1]])
            travel_start = current_time
            travel_end = travel_start + datetime.timedelta(hours=travel_time)
            timetable.append(
                {
                    "type": "travel",
                    "from_spot_id": spots[route[0]].tourist_spot_id,
                    "from_spot_name": spots[route[0]].name,
                    "to_spot_id": spots[route[1]].tourist_spot_id,
                    "to_spot_name": spots[route[1]].name,
                    "start_time": travel_start.time(),
                    "end_time": travel_end.time(),
                    "duration": travel_time,
                }
            )
            return timetable

        # 기존 로직 (route 길이가 3 이상인 경우)
        for i, (spot_idx, visit_time, travel_time) in enumerate(detailed_times):
            spot = spots[spot_idx]
            if i > 0:
                visit_start = current_time
                visit_end = visit_start + datetime.timedelta(hours=visit_time)
                timetable.append(
                    {
                        "type": "visit",
                        "spot_id": spot.tourist_spot_id,
                        "spot_name": spot.name,
                        "start_time": visit_start.time(),
                        "end_time": visit_end.time(),
                        "duration": visit_time,
                    }
                )
                current_time = visit_end
            if i < len(detailed_times) - 1:
                next_spot_idx = route[i + 1]
                next_spot = spots[next_spot_idx]
                travel_start = current_time
                travel_end = travel_start + datetime.timedelta(hours=travel_time)
                timetable.append(
                    {
                        "type": "travel",
                        "from_spot_id": spot.tourist_spot_id,
                        "from_spot_name": spot.name,
                        "to_spot_id": next_spot.tourist_spot_id,
                        "to_spot_name": next_spot.name,
                        "start_time": travel_start.time(),
                        "end_time": travel_end.time(),
                        "duration": travel_time,
                    }
                )
                current_time = travel_end
        return timetable

    def check_time_constraints(
        self,
        spots: List[TouristSpot],
        route: List[int],
        max_duration: float,
        start_time: datetime.time = datetime.time(9),
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
        """
        시간 제약 조건 확인 및 일정표 생성

        Args:
            spots: 관광지 목록
            route: 경로 인덱스 목록
            max_duration: 최대 소요 시간 (시간)
            start_time: 시작 시간

        Returns:
            (제약 조건 만족 여부, 일정표)
        """
        if not route or len(route) < 2:
            return False, None

        # 이동 시간과 관광 시간 계산
        travel_time, visit_time, _ = self.calculate_route_times(spots, route)
        total_time = travel_time + visit_time

        # 제약 조건 확인
        if total_time > max_duration:
            return False, None

        # 일정표 생성
        timetable = self.generate_timetable(spots, route, start_time)

        return True, timetable

    def adjust_route_for_time_constraints(
        self,
        spots: List[TouristSpot],
        route: List[int],
        max_duration: float,
        priority_scores: Dict[int, float],
    ) -> List[int]:
        """
        시간 제약 조건에 맞게 경로 조정

        Args:
            spots: 관광지 목록
            route: 경로 인덱스 목록
            max_duration: 최대 소요 시간 (시간)
            priority_scores: 관광지 ID를 키로, 우선순위 점수를 값으로 하는 딕셔너리

        Returns:
            조정된 경로 인덱스 목록
        """
        if not route or len(route) < 2:
            return route

        # 이동 시간과 관광 시간 계산
        travel_time, visit_time, detailed_times = self.calculate_route_times(
            spots, route
        )
        total_time = travel_time + visit_time

        # 제약 조건 만족 시 그대로 반환
        if total_time <= max_duration:
            return route

        # 시작 지점과 종료 지점 (동일)
        start_idx = route[0]

        # 중간 지점들의 우선순위 계산
        middle_points = route[1:-1]
        priorities = []

        for idx in middle_points:
            spot_id = spots[idx].tourist_spot_id
            score = priority_scores.get(spot_id, 0.0)
            priorities.append((idx, score))

        # 우선순위 내림차순 정렬
        priorities.sort(key=lambda x: x[1], reverse=True)

        # 시간 제약 조건을 만족할 때까지 우선순위가 낮은 지점 제거
        adjusted_route = [start_idx]

        for idx, _ in priorities:
            # 임시로 이 지점을 추가
            temp_route = adjusted_route + [idx, start_idx]

            # 이동 시간과 관광 시간 계산
            temp_travel_time, temp_visit_time, _ = self.calculate_route_times(
                spots, temp_route
            )
            temp_total_time = temp_travel_time + temp_visit_time

            # 시간 제약 조건 확인
            if temp_total_time <= max_duration:
                adjusted_route.append(idx)

        # 시작 지점으로 돌아가기
        adjusted_route.append(start_idx)

        return adjusted_route
