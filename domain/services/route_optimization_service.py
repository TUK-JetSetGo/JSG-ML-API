"""
MILP를 이용한 세부 동선 생성 서비스 로직
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, invalid-name, W0612

from typing import Optional

from infrastructure.adapters.ml.enhanced_ptppp_model import EnhancedPTPPPModel


class RouteOptimizationService:
    @staticmethod
    def calculate_distance_matrix(spots):
        """
        Calculate a distance matrix (in km) between spots using Euclidean distance in degrees
        multiplied by the km-per-degree factor.
        """
        km_per_degree = 111.31949079327357
        n = len(spots)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                else:
                    deg = spots[i].coordinate.distance_to(spots[j].coordinate)
                    row.append(deg * km_per_degree)
            matrix.append(row)
        return matrix

    @staticmethod
    def optimize_single_day_route(
        spots,
        start_spot_index,
        base_scores,
        priority_scores,
        max_distance,
        max_duration,
        max_places,
        must_visit_indices=None,
        transport_speed_kmh=40.0,
        prev_route_end_index: Optional[int] = None,
    ):
        if not spots:
            return [], 0.0, 0.0

        distance_matrix = RouteOptimizationService.calculate_distance_matrix(spots)
        model = EnhancedPTPPPModel(
            distance_matrix=distance_matrix,
            base_scores=base_scores,
            priority_scores=priority_scores,
            max_distance=max_distance,
            max_duration=max_duration,
            max_places=max_places,
            transport_speed_kmh=transport_speed_kmh,
        )
        prev_day_end_provided = prev_route_end_index is not None
        route, total_distance, total_duration = model._optimize_single_day_route(
            n=len(spots),
            K=max_places,
            prev_day_end_provided=prev_day_end_provided,
        )
        return route, total_distance, total_duration

    @staticmethod
    def optimize_route(
        spots,
        start_spot_index,
        base_scores,
        priority_scores,
        max_distance,
        max_duration,
        max_places,
        must_visit_indices=None,
        transport_speed_kmh=40.0,
        prev_route_end_index: Optional[int] = None,
    ):
        """
        Optimize a single-day route using EnhancedPTPPPModel.optimize_route.
        """
        model = EnhancedPTPPPModel()
        return model.optimize_route(
            spots=spots,
            start_spot_index=start_spot_index,
            base_scores=base_scores,
            priority_scores=priority_scores,
            max_distance=max_distance,
            max_duration=max_duration,
            max_places=max_places,
            must_visit_indices=(
                must_visit_indices if must_visit_indices is not None else []
            ),
            transport_speed_kmh=transport_speed_kmh,
        )

    @staticmethod
    def optimize_multi_day_route(
        clusters,
        user_profile,
        num_days,
        max_places_per_day,
        daily_start_points,
        daily_max_distance,
        daily_max_duration,
        base_scores,
        priority_scores,
    ):
        results = []
        prev_end_index = None
        for day_idx in range(num_days):
            spots = clusters[day_idx]
            start_idx = daily_start_points[day_idx]
            max_dist = daily_max_distance[day_idx]
            max_dur = daily_max_duration[day_idx]
            bs = base_scores[day_idx]
            ps = priority_scores[day_idx]

            route, dist, dur = RouteOptimizationService.optimize_single_day_route(
                spots=spots,
                start_spot_index=start_idx,
                base_scores=bs,
                priority_scores=ps,
                max_distance=max_dist,
                max_duration=max_dur,
                max_places=max_places_per_day,
                must_visit_indices=None,
                transport_speed_kmh=user_profile.transport_speed_kmh,
                prev_route_end_index=prev_end_index,
            )
            results.append((route, dist, dur))
            if len(route) >= 2:
                prev_end_index = route[-2]
            else:
                prev_end_index = None
        return results
