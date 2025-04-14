"""
MILP를 이용한 세부 동선 생성 서비스 로직
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, invalid-name, W0612

from typing import Dict, List, Optional, Tuple

import pulp

from build.lib.domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.value_objects.coordinate import Coordinate


class RouteOptimizationService:
    @staticmethod
    def calculate_distance_matrix(spots: List[TouristSpot]) -> List[List[float]]:
        """
        관광지 간 거리 행렬 계산

        Args:
            spots: 관광지 목록

        Returns:
            거리 행렬
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

                distance_matrix[i][j] = coord_i.distance_to(coord_j)
        return distance_matrix

    @staticmethod
    def optimize_single_day_route(
        spots: List[TouristSpot],
        start_spot_index: int,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        max_distance: float,
        max_duration: float,
        max_places: int,
        must_visit_indices: List[int] = None,
        transport_speed_kmh: float = 40.0,
    ) -> Tuple[List[int], float, float]:
        """
        단일 일자 경로 최적화 (PTPPP)

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

        Returns:
            최적 경로 인덱스 목록, 총 이동 거리, 총 소요 시간
        """

        if not spots:
            return [], 0, 0

        unique_spots = []
        unique_indices = []
        seen_ids = set()

        for i, spot in enumerate(spots):
            if spot.id not in seen_ids:
                unique_spots.append(spot)
                unique_indices.append(i)
                seen_ids.add(spot.id)

        # 시작지점이 중복 제거 이후에도 존재하는지 확인하고, 그렇지 않다면 시작지점을 리스트 맨 앞으로 이동.
        if start_spot_index not in unique_indices:
            unique_spots.insert(0, spots[start_spot_index])
            unique_indices.insert(0, start_spot_index)

        else:
            start_idx = unique_indices.index(start_spot_index)
            unique_spots.insert(0, unique_spots.pop(start_idx))
            unique_indices.insert(0, unique_indices.pop(start_idx))

        n = len(unique_spots)
        K = min(n - 1, max_places - 1)

        idx_to_id = {i: unique_spots[i].id for i in range(n)}
        id_to_idx = {v: k for k, v in idx_to_id.items()}

        cost_matrix = RouteOptimizationService.calculate_distance_matrix(unique_spots)

        prob = pulp.LpProblem("PTPPP_Day", pulp.LpMaximize)

        # 결정 변수 정의
        X = pulp.LpVariable.dicts(
            name="X",
            indices=[(i, j) for i in range(n) for j in range(n) if i != j],
            cat=pulp.LpBinary,
        )

        Y = pulp.LpVariable.dicts(
            name="Y",
            indices=[(k, i) for k in range(1, K + 1) for i in range(1, n)],
            cat=pulp.LpBinary,
        )

        Z = pulp.LpVariable.dicts(
            name="Z", indices=[i for i in range(1, n)], lowBound=0
        )

        # 목적 함수 정의
        prize_terms = []
        for i in range(1, n):
            spot_id = idx_to_id[i]
            base_val = base_scores.get(spot_id, 0.0)
            base_expr = base_val * pulp.lpSum([Y[k, i] for k in range(1, K + 1)])
            prize_terms.append(base_expr)

        cost_terms = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cost_terms.append(cost_matrix[i][j] * X[(i, j)])

        prob += pulp.lpSum(prize_terms) - pulp.lpSum(cost_terms)

        # 제약 조건 시작

        prob += pulp.lpSum([X[(0, j)] for j in range(1, n)]) <= 1, "Start_leaving"
        prob += pulp.lpSum([X[(i, 0)] for i in range(1, n)]) <= 1, "End_return"

        for i in range(1, n):
            inbound = pulp.lpSum([X[(h, i)] for h in range(n) if h != i])
            outbound = pulp.lpSum([X[(i, h)] for h in range(n) if h != i])
            visited = pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)])
            prob += inbound == visited, f"InboundNode_{i}"
            prob += outbound == visited, f"OutboundNode_{i}"

        for i in range(1, n):
            prob += (
                pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)]) <= 1,
                f"OneOrder_{i}",
            )

        for k in range(1, K + 1):
            prob += pulp.lpSum([Y[(k, i)] for i in range(1, n)]) <= 1, f"OrderCap_{k}"

        for k in range(1, K):
            prob += (
                pulp.lpSum([Y[(k, i)] for i in range(1, n)])
                >= pulp.lpSum([Y[(k + 1, i)] for i in range(1, n)])
            ), f"NoGap_{k}"

        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                for k in range(2, K + 1):
                    prob += (
                        X[(i, j)] >= Y[(k - 1, i)] + Y[(k, j)] - 1,
                        f"Link_{i}_{j}_k{k}",
                    )

        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                prob += Z[i] - Z[j] + (n + 1) * X[(i, j)] <= n, f"Subtour_{i}_{j}"

        if must_visit_indices:
            for idx in must_visit_indices:
                if idx in unique_indices and idx != start_spot_index:
                    i_idx = unique_indices.index(idx)
                    if i_idx != 0:  # 시작 지점이 아닌 경우
                        prob += (
                            pulp.lpSum([Y[(k, i_idx)] for k in range(1, K + 1)]) == 1,
                            f"MustVisit_{idx}",
                        )

        distance_expr = pulp.lpSum(
            [
                cost_matrix[i][j] * X[(i, j)]
                for i in range(n)
                for j in range(n)
                if i != j
            ]
        )
        prob += distance_expr <= max_distance, "MaxDistance"
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
        prob += (
            pulp.lpSum([Y[(k, i)] for i in range(1, n) for k in range(1, K + 1)])
            <= (max_places - 1)
        ), "MaxPlaces"

        # 문제 해결
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] not in ["Optimal", "Feasible"]:
            return [], 0.0, 0.0

        visited_sequence = []
        for k in range(1, K + 1):
            for i in range(1, n):
                val = pulp.value(Y[(k, i)])
                if val and val > 0.5:
                    visited_sequence.append(unique_indices[i])

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

    @staticmethod
    def optimize_multi_day_route(
        clusters: Dict[int, List[TouristSpot]],
        user_profile: UserProfile,
        num_days: int,
        max_places_per_day: int,
        daily_start_points: List[Optional[int]],
        daily_max_distance: float,
        daily_max_duration: float,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        transport_speed_kmh: float = 40.0,
    ) -> List[Tuple[List[int], float, float]]:
        """
        다중 일자 경로 최적화

        Args:
            clusters: 클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
            user_profile: 사용자 프로필
            num_days: 일수
            max_places_per_day: 일일 최대 방문 장소 수
            daily_start_points: 일별 시작 지점 인덱스 목록
            daily_max_distance: 일일 최대 이동 거리 (km)
            daily_max_duration: 일일 최대 소요 시간 (시간)
            base_scores: 관광지 ID를 키로, 기본 점수를 값으로 하는 딕셔너리
            priority_scores: 우선순위 레벨을 첫 번째 키로, 관광지 ID를 두 번째 키로, 점수를 값으로 하는 중첩 딕셔너리
            transport_speed_kmh: 이동 속도 (km/h)

        Returns:
            일별 (최적 경로 인덱스 목록, 총 이동 거리, 총 소요 시간) 튜플 목록
        """
        # 클러스터 점수 계산
        cluster_scores = {}
        for cluster_id, spots in clusters.items():
            # 클러스터 내 관광지의 평균 점수
            spot_scores = [base_scores.get(spot.id, 0.0) for spot in spots]
            avg_score = sum(spot_scores) / len(spot_scores) if spot_scores else 0.0

            # 반드시 방문해야 하는 관광지가 있는 경우 가중치 부여
            must_visit_bonus = sum(
                1000.0 for spot in spots if spot.id in user_profile.must_visit_list
            )

            # 클러스터 점수 = 평균 점수 + 반드시 방문 보너스
            cluster_scores[cluster_id] = avg_score + must_visit_bonus

        # 클러스터 점수 기준 내림차순 정렬
        sorted_clusters = sorted(
            clusters.keys(), key=lambda cid: cluster_scores.get(cid, 0.0), reverse=True
        )

        # 일수보다 클러스터가 적은 경우 처리
        if len(sorted_clusters) < num_days:
            sorted_clusters.extend(
                [sorted_clusters[0]] * (num_days - len(sorted_clusters))
            )

        # 일별 최적 경로 계산
        daily_routes = []
        for day_idx in range(num_days):
            cluster_id = sorted_clusters[day_idx]
            spots = clusters[cluster_id]

            # 시작 지점 결정
            start_spot_idx = 0
            if (
                day_idx < len(daily_start_points)
                and daily_start_points[day_idx] is not None
            ):
                # 지정된 시작 지점이 있는 경우
                start_id = daily_start_points[day_idx]
                for i, spot in enumerate(spots):
                    if spot.id == start_id:
                        start_spot_idx = i
                        break

            # 반드시 방문해야 하는 관광지 인덱스 목록
            must_visit_indices = []
            for i, spot in enumerate(spots):
                if spot.id in user_profile.must_visit_list:
                    must_visit_indices.append(i)

            # 일별 최적 경로 계산
            route, dist, dur = RouteOptimizationService.optimize_single_day_route(
                spots=spots,
                start_spot_index=start_spot_idx,
                base_scores=base_scores,
                priority_scores=priority_scores,
                max_distance=daily_max_distance,
                max_duration=daily_max_duration,
                max_places=max_places_per_day,
                must_visit_indices=must_visit_indices,
                transport_speed_kmh=transport_speed_kmh,
            )

            daily_routes.append((route, dist, dur))

        return daily_routes
