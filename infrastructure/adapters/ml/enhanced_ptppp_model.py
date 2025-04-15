"""
개선된 PTPPP 모델 모듈 - 일자별 연결성 고려
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.value_objects.coordinate import Coordinate


class EnhancedPTPPPModel:
    """일자별 연결성을 고려한 개선된 PTPPP 모델"""

    def __init__(self):
        """초기화"""
        pass

    def calculate_distance_matrix(self, spots: List[TouristSpot]) -> List[List[float]]:
        """
        관광지 간 거리 행렬 계산

        Args:
            spots: 관광지 목록

        Returns:
            거리 행렬 (2차원 리스트)
        """
        n = len(spots)
        distance_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                coord_i = Coordinate(
                    spots[i].coordinate.latitude, spots[i].coordinate.longitude
                )
                coord_j = Coordinate(
                    spots[j].coordinate.latitude, spots[j].coordinate.longitude
                )

                # 하버사인 공식으로 거리 계산
                distance_matrix[i][j] = coord_i.distance_to(coord_j)

        return distance_matrix

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
        transport_speed_kmh: float,
    ) -> Tuple[List[int], float, float]:
        """
        단일 일자 경로 최적화를 수행하는 메서드의 래퍼입니다.

        Args:
            spots: 관광지 목록
            start_spot_index: 시작 관광지 인덱스
            base_scores: 관광지 ID를 키로, 기본 점수를 값으로 하는 딕셔너리
            priority_scores: 우선순위 점수를 담은 중첩 딕셔너리 (예: {order: {spot.id: score}})
            max_distance: 일일 최대 이동 거리 (km)
            max_duration: 일일 최대 소요 시간 (시간)
            max_places: 일일 최대 방문 장소 수
            must_visit_indices: 반드시 방문해야 하는 관광지 인덱스 목록
            transport_speed_kmh: 이동 속도 (km/h)

        Returns:
            (최적 경로 인덱스 목록, 총 이동 거리, 총 소요 시간)
        """
        # prev_day_end은 단일 일자 최적화를 위해 None, continuity_weight은 기본 0.5 사용
        return self._optimize_single_day_route(
            spots=spots,
            start_spot_index=start_spot_index,
            base_scores=base_scores,
            priority_scores=priority_scores,
            max_distance=max_distance,
            max_duration=max_duration,
            max_places=max_places,
            must_visit_indices=must_visit_indices,
            transport_speed_kmh=transport_speed_kmh,
            prev_day_end=None,
            continuity_weight=0.5,
        )

    def optimize_multi_day_route(
        self,
        daily_spots: List[List[TouristSpot]],
        user_profile: UserProfile,
        daily_start_indices: List[int],
        daily_max_distance: float,
        daily_max_duration: float,
        max_places_per_day: int,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        transport_speed_kmh: float = 40.0,
        continuity_weight: float = 0.5,
    ) -> List[Tuple[List[int], float, float]]:
        """
        일자별 연결성을 고려한 다중 일자 경로 최적화

        Args:
            daily_spots: 일별 관광지 목록
            user_profile: 사용자 프로필
            daily_start_indices: 일별 시작 지점 인덱스 목록
            daily_max_distance: 일일 최대 이동 거리 (km)
            daily_max_duration: 일일 최대 소요 시간 (시간)
            max_places_per_day: 일일 최대 방문 장소 수
            base_scores: 관광지 ID를 키로, 기본 점수를 값으로 하는 딕셔너리
            priority_scores: 우선순위 레벨을 첫 번째 키로, 관광지 ID를 두 번째 키로, 점수를 값으로 하는 중첩 딕셔너리
            transport_speed_kmh: 이동 속도 (km/h)
            continuity_weight: 일자 간 연속성 가중치 (0.0 ~ 1.0)

        Returns:
            일별 (최적 경로 인덱스 목록, 총 이동 거리, 총 소요 시간) 튜플 목록
        """
        num_days = len(daily_spots)
        if num_days == 0:
            return []

        # 각 일자별 거리 행렬 계산
        daily_distance_matrices = [
            self.calculate_distance_matrix(spots) for spots in daily_spots
        ]

        # 각 일자별 ID-인덱스 매핑
        daily_id_to_idx = []
        daily_idx_to_id = []

        for day_spots in daily_spots:
            id_to_idx = {spot.tourist_spot_id: i for i, spot in enumerate(day_spots)}
            idx_to_id = {i: spot.tourist_spot_id for i, spot in enumerate(day_spots)}
            daily_id_to_idx.append(id_to_idx)
            daily_idx_to_id.append(idx_to_id)

        # 일자별 최적 경로 계산 (첫째 날)
        first_day_route, first_day_dist, first_day_dur = (
            self._optimize_single_day_route(
                spots=daily_spots[0],
                start_spot_index=daily_start_indices[0],
                base_scores=base_scores,
                priority_scores=priority_scores,
                max_distance=daily_max_distance,
                max_duration=daily_max_duration,
                max_places=max_places_per_day,
                must_visit_indices=self._get_must_visit_indices(
                    daily_spots[0], user_profile
                ),
                transport_speed_kmh=transport_speed_kmh,
                prev_day_end=None,  # 첫째 날은 이전 일자 종료 지점 없음
                continuity_weight=continuity_weight,
            )
        )

        daily_routes = [(first_day_route, first_day_dist, first_day_dur)]

        # 이전 일자 종료 지점 (다음 날 시작 지점과 연결성 고려)
        prev_day_end = None
        if first_day_route:
            last_idx = first_day_route[
                -2
            ]  # 마지막 방문 지점 (시작 지점으로 돌아가기 전)
            prev_day_end = daily_spots[0][last_idx]

        # 나머지 일자 최적 경로 계산
        for day in range(1, num_days):
            day_route, day_dist, day_dur = self._optimize_single_day_route(
                spots=daily_spots[day],
                start_spot_index=daily_start_indices[day],
                base_scores=base_scores,
                priority_scores=priority_scores,
                max_distance=daily_max_distance,
                max_duration=daily_max_duration,
                max_places=max_places_per_day,
                must_visit_indices=self._get_must_visit_indices(
                    daily_spots[day], user_profile
                ),
                transport_speed_kmh=transport_speed_kmh,
                prev_day_end=prev_day_end,  # 이전 일자 종료 지점
                continuity_weight=continuity_weight,
            )

            daily_routes.append((day_route, day_dist, day_dur))

            # 이전 일자 종료 지점 업데이트
            if day_route:
                last_idx = day_route[-2]  # 마지막 방문 지점 (시작 지점으로 돌아가기 전)
                prev_day_end = daily_spots[day][last_idx]

        return daily_routes

    def _get_must_visit_indices(
        self, spots: List[TouristSpot], user_profile: UserProfile
    ) -> List[int]:
        """
        반드시 방문해야 하는 관광지 인덱스 목록 반환

        Args:
            spots: 관광지 목록
            user_profile: 사용자 프로필

        Returns:
            반드시 방문해야 하는 관광지 인덱스 목록
        """
        must_visit_indices = []
        for i, spot in enumerate(spots):
            if spot.tourist_spot_id in user_profile.must_visit_list:
                must_visit_indices.append(i)
        return must_visit_indices

    def _optimize_single_day_route(
        self,
        spots: List[TouristSpot],
        start_spot_index: int,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        max_distance: float,
        max_duration: float,
        max_places: int,
        must_visit_indices: List[int] = None,
        transport_speed_kmh: float = 40.0,
        prev_day_end: Optional[TouristSpot] = None,
        continuity_weight: float = 0.5,
    ) -> Tuple[List[int], float, float]:
        """
        단일 일자 경로 최적화 (PTPPP 모델 적용)

        Args:
            spots: 관광지 목록
            start_spot_index: 시작 관광지 인덱스
            base_scores: 관광지 ID를 키로, 기본 점수를 값으로 하는 딕셔너리
            priority_scores: 우선순위 레벨을 첫 번째 키로, 관광지 ID를 두 번째 키로, 점수를 값으로 하는 중첩 딕셔너리
            max_distance: 최대 이동 거리 (km)
            max_duration: 최대 소요 시간 (시간)
            max_places: 최대 방문 장소 수
            must_visit_indices: 반드시 방문해야 하는 관광지 인덱스 목록
            transport_speed_kmh: 이동 속도 (km/h)
            prev_day_end: 이전 일자 종료 관광지 (선택 사항)
            continuity_weight: 일자 간 연속성 가중치 (0.0 ~ 1.0)

        Returns:
            (최적 경로 인덱스 목록, 총 이동 거리, 총 소요 시간)
        """
        if not spots:
            return [], 0.0, 0.0

        # 중복 제거
        unique_spots = []
        unique_indices = []
        seen_ids = set()

        for i, spot in enumerate(spots):
            if spot.tourist_spot_id not in seen_ids:
                unique_spots.append(spot)
                unique_indices.append(i)
                seen_ids.add(spot.tourist_spot_id)

        # 시작 지점이 중복 제거 후에도 존재하는지 확인
        if start_spot_index not in unique_indices:
            unique_spots.insert(0, spots[start_spot_index])
            unique_indices.insert(0, start_spot_index)
        else:
            # 시작 지점을 리스트의 맨 앞으로 이동
            start_idx = unique_indices.index(start_spot_index)
            unique_spots.insert(0, unique_spots.pop(start_idx))
            unique_indices.insert(0, unique_indices.pop(start_idx))

        # 방문 장소 수 제한
        n = len(unique_spots)
        K = min(n - 1, max_places - 1)  # 시작 지점 제외

        # 인덱스-ID 매핑
        idx_to_id = {i: unique_spots[i].id for i in range(n)}
        id_to_idx = {v: k for k, v in idx_to_id.items()}

        # 거리 행렬 계산
        cost_matrix = self.calculate_distance_matrix(unique_spots)

        # 이전 일자 종료 지점과의 연속성 점수 계산
        continuity_scores = np.zeros(n)
        if prev_day_end is not None:
            prev_end_coord = Coordinate(
                prev_day_end.coordinate.latitude, prev_day_end.coordinate.longitude
            )
            for i, spot in enumerate(unique_spots):
                if i == 0:  # 시작 지점은 제외
                    continue
                spot_coord = Coordinate(
                    spot.coordinate.latitude, spot.coordinate.longitude
                )
                distance = prev_end_coord.distance_to(spot_coord)
                # 거리가 가까울수록 높은 점수 (최대 100점)
                continuity_scores[i] = 100.0 * math.exp(
                    -distance / 10.0
                )  # 10km 거리에서 약 36.8점

        # MILP 문제 정의
        prob = pulp.LpProblem("PTPPP_Day", pulp.LpMaximize)

        # 결정 변수 정의
        # X[i,j] = 1 if edge (i,j) is in the path, 0 otherwise
        X = pulp.LpVariable.dicts(
            "X",
            [(i, j) for i in range(n) for j in range(n) if i != j],
            cat=pulp.LpBinary,
        )

        # Y[k,i] = 1 if vertex i is visited at position k, 0 otherwise
        Y = pulp.LpVariable.dicts(
            "Y",
            [(k, i) for k in range(1, K + 1) for i in range(1, n)],
            cat=pulp.LpBinary,
        )

        # Z[i] = position of vertex i in the path (for subtour elimination)
        Z = pulp.LpVariable.dicts(
            "Z", [i for i in range(1, n)], lowBound=0, cat=pulp.LpContinuous
        )

        # 목적 함수: 점수 최대화 - 비용 최소화 + 연속성 점수
        prize_terms = []
        for i in range(1, n):
            spot_id = idx_to_id[i]
            base_val = base_scores.get(spot_id, 0.0)
            base_expr = base_val * pulp.lpSum([Y[k, i] for k in range(1, K + 1)])
            prize_terms.append(base_expr)

            for k in range(1, K + 1):
                prio_val = priority_scores.get(k, {}).get(spot_id, 0.0)
                prize_terms.append(prio_val * Y[k, i])

                # 연속성 점수 추가 (첫 번째 방문 지점에 대해서만)
                if k == 1 and continuity_scores[i] > 0:
                    prize_terms.append(
                        continuity_weight * continuity_scores[i] * Y[k, i]
                    )

        cost_terms = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cost_terms.append(cost_matrix[i][j] * X[(i, j)])

        prob += pulp.lpSum(prize_terms) - pulp.lpSum(cost_terms), "Max_Prize_minus_Cost"

        # 제약 조건
        # 시작 지점에서 최대 1개의 간선 출발
        prob += pulp.lpSum([X[(0, j)] for j in range(1, n)]) <= 1, "Start_leaving"

        # 시작 지점으로 최대 1개의 간선 도착
        prob += pulp.lpSum([X[(i, 0)] for i in range(1, n)]) <= 1, "End_return"

        # 각 정점의 진입 간선 수 = 방문 여부
        for i in range(1, n):
            inbound = pulp.lpSum([X[(h, i)] for h in range(n) if h != i])
            outbound = pulp.lpSum([X[(i, h)] for h in range(n) if h != i])
            visited = pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)])
            prob += inbound == visited, f"InboundNode_{i}"
            prob += outbound == visited, f"OutboundNode_{i}"

        # 각 정점은 최대 1번 방문
        for i in range(1, n):
            prob += (
                pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)]) <= 1,
                f"OneOrder_{i}",
            )

        # 각 순서에는 최대 1개의 정점 할당
        for k in range(1, K + 1):
            prob += pulp.lpSum([Y[(k, i)] for i in range(1, n)]) <= 1, f"OrderCap_{k}"

        # 순서 간 갭 없음 (k번째 정점이 있으면 k-1번째 정점도 있어야 함)
        for k in range(1, K):
            prob += (
                pulp.lpSum([Y[(k, i)] for i in range(1, n)])
                >= pulp.lpSum([Y[(k + 1, i)] for i in range(1, n)])
            ), f"NoGap_{k}"

        # 순서가 연속된 정점 간에는 간선이 있어야 함
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                for k in range(2, K + 1):
                    prob += (
                        X[(i, j)] >= Y[(k - 1, i)] + Y[(k, j)] - 1,
                        f"Link_{i}_{j}_k{k}",
                    )

        # 서브투어 제거 제약
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                prob += Z[i] - Z[j] + (n + 1) * X[(i, j)] <= n, f"Subtour_{i}_{j}"

        # 반드시 방문해야 하는 관광지 제약
        if must_visit_indices:
            for idx in must_visit_indices:
                if idx in unique_indices and idx != start_spot_index:
                    i_idx = unique_indices.index(idx)
                    if i_idx != 0:  # 시작 지점이 아닌 경우
                        prob += (
                            pulp.lpSum([Y[(k, i_idx)] for k in range(1, K + 1)]) == 1,
                            f"MustVisit_{idx}",
                        )

        # 최대 이동 거리 제약
        distance_expr = pulp.lpSum(
            [
                cost_matrix[i][j] * X[(i, j)]
                for i in range(n)
                for j in range(n)
                if i != j
            ]
        )
        prob += distance_expr <= max_distance, "MaxDistance"

        # 최대 소요 시간 제약
        visit_time_terms = []
        for i in range(1, n):
            # 각 관광지 방문 시간
            visit_duration = unique_spots[i].average_visit_duration
            visit_time_terms.append(
                visit_duration * pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)])
            )

        travel_time_terms = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist_ij = cost_matrix[i][j]
                travel_time_terms.append((dist_ij / transport_speed_kmh) * X[(i, j)])

        total_time_expr = pulp.lpSum(visit_time_terms) + pulp.lpSum(travel_time_terms)
        prob += total_time_expr <= max_duration, "MaxDuration"

        # 최대 방문 장소 수 제약
        prob += (
            pulp.lpSum([Y[(k, i)] for i in range(1, n) for k in range(1, K + 1)])
            <= (max_places - 1)
        ), "MaxPlaces"

        # 문제 해결
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # 최적해가 없는 경우
        if pulp.LpStatus[prob.status] not in ["Optimal", "Feasible"]:
            return [], 0.0, 0.0

        # 방문 순서 추출
        visited_sequence = []
        for k in range(1, K + 1):
            for i in range(1, n):
                val = pulp.value(Y[(k, i)])
                if val and val > 0.5:
                    visited_sequence.append(unique_indices[i])

        # 전체 경로 (시작 지점 포함)
        full_route = [start_spot_index] + visited_sequence + [start_spot_index]

        # 총 이동 거리 계산
        total_dist = 0.0
        for i in range(len(full_route) - 1):
            idx_a = full_route[i]
            idx_b = full_route[i + 1]
            spot_a = spots[idx_a]
            spot_b = spots[idx_b]
            coord_a = Coordinate(
                spot_a.coordinate.latitude, spot_a.coordinate.longitude
            )
            coord_b = Coordinate(
                spot_b.coordinate.latitude, spot_b.coordinate.longitude
            )
            dist_ab = coord_a.distance_to(coord_b)
            total_dist += dist_ab

        # 총 소요 시간 계산
        total_travel_time = total_dist / transport_speed_kmh
        total_visit_time = sum(
            spots[idx].average_visit_duration for idx in visited_sequence
        )
        total_dur = total_travel_time + total_visit_time

        return full_route, total_dist, total_dur
