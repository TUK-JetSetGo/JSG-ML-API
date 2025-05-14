"""
Enhanced Tabu Search Model for PTPPP, with unified interface and internal handling of transport speed and continuity.
Based on "Optimization approaches to support the planning and analysis of travel itineraries" by Da Silva et al. (2018).
Specifically, the Tabu Search algorithm described in Section 4.

Refactored to use domain.entities.TouristSpot, domain.entities.UserProfile, and domain.value_objects.Coordinate.
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile

logger = logging.getLogger(__name__)

DEFAULT_TRANSPORT_SPEEDS = {
    "car": 60.0,
    "public_transport": 40.0,
    "walk": 5.0,
    "default": 40.0,
}

DEFAULT_CONTINUITY_WEIGHT = 0.5


class EnhancedTabuSearchModel:
    def _get_spot_id(self, spot: TouristSpot) -> int:
        # Ensure compatibility with DummyTouristSpot which might use 'id' instead of 'tourist_spot_id'
        return getattr(
            spot, "tourist_spot_id", getattr(spot, "id", -1)
        )  # Return -1 or raise error if no ID

    def __init__(self):
        pass

    def _get_transport_speed(self, user_profile: Optional[UserProfile]) -> float:
        if user_profile and user_profile.preferred_transport:
            return DEFAULT_TRANSPORT_SPEEDS.get(
                user_profile.preferred_transport.lower(),
                DEFAULT_TRANSPORT_SPEEDS["default"],
            )
        return DEFAULT_TRANSPORT_SPEEDS["default"]

    def calculate_distance_matrix(self, spots: List[TouristSpot]) -> List[List[float]]:
        n = len(spots)
        distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Handle potential None coordinates gracefully, though ideally spots should always have coordinates
                if spots[i].coordinate and spots[j].coordinate:
                    distance_matrix[i][j] = spots[i].coordinate.distance_to(
                        spots[j].coordinate
                    )
                else:
                    logger.warning(
                        f"Missing coordinate for spot {self._get_spot_id(spots[i])} or {self._get_spot_id(spots[j])}. Setting distance to infinity."
                    )
                    distance_matrix[i][j] = float("inf")  # Or handle as an error
        return distance_matrix

    def _calculate_route_properties(
        self,
        route_indices: List[int],
        spots: List[TouristSpot],
        distance_matrix: List[List[float]],
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        transport_speed_kmh: float,
        prev_day_end: Optional[TouristSpot],
        continuity_weight: float,
    ) -> Tuple[float, float, float]:
        if not route_indices or len(route_indices) < 2:
            return -float("inf"), 0.0, 0.0

        total_distance = 0.0
        total_visit_time = 0.0
        current_score = 0.0

        for i in range(len(route_indices) - 1):
            u, v = route_indices[i], route_indices[i + 1]
            if 0 <= u < len(distance_matrix) and 0 <= v < len(distance_matrix[u]):
                total_distance += distance_matrix[u][v]
            else:
                logger.error(f"Invalid indices for distance_matrix: u={u}, v={v}")
                return -float("inf"), float("inf"), float("inf")  # Indicate error

        total_travel_time = (
            total_distance / transport_speed_kmh
            if transport_speed_kmh > 0
            else float("inf")
        )
        visited_spots_in_order_indices = route_indices[1:-1]

        for spot_idx in visited_spots_in_order_indices:
            if 0 <= spot_idx < len(spots):
                spot_obj = spots[spot_idx]
                total_visit_time += spot_obj.average_visit_duration
                current_score += base_scores.get(self._get_spot_id(spot_obj), 0.0)
            else:
                logger.error(f"Invalid spot_idx: {spot_idx}")
                return -float("inf"), total_distance, float("inf")  # Indicate error

        for k, spot_idx in enumerate(visited_spots_in_order_indices):
            order = k + 1
            if 0 <= spot_idx < len(spots):
                current_score += priority_scores.get(order, {}).get(
                    self._get_spot_id(spots[spot_idx]), 0.0
                )
            else:
                logger.error(
                    f"Invalid spot_idx in priority score calculation: {spot_idx}"
                )
                # Continue, as this might not be a fatal error for scoring

        if prev_day_end and visited_spots_in_order_indices:
            first_spot_idx = visited_spots_in_order_indices[0]
            if 0 <= first_spot_idx < len(spots):
                first_spot_obj = spots[first_spot_idx]
                if prev_day_end.coordinate and first_spot_obj.coordinate:
                    dist_to_first = prev_day_end.coordinate.distance_to(
                        first_spot_obj.coordinate
                    )
                    current_score += continuity_weight * (
                        100.0 * math.exp(-dist_to_first / 10.0)
                    )
                else:
                    logger.warning("Missing coordinates for continuity calculation.")
            else:
                logger.error(f"Invalid first_spot_idx for continuity: {first_spot_idx}")

        total_duration = total_travel_time + total_visit_time
        return current_score, total_distance, total_duration

    def _is_route_feasible(
        self,
        route_indices: List[int],
        total_distance: float,
        total_duration: float,
        max_distance: float,
        max_duration: float,
        max_places: int,
        must_visit_spot_indices_set: Set[int],
        start_spot_index: int,
    ) -> bool:
        if (
            not route_indices
            or route_indices[0] != start_spot_index
            or route_indices[-1] != start_spot_index
        ):
            return False
        actual_visited_spots_indices = set(route_indices[1:-1])
        if total_distance > max_distance + 1e-9:
            return False
        if total_duration > max_duration + 1e-9:
            return False
        if len(actual_visited_spots_indices) > max_places:
            return False
        if not must_visit_spot_indices_set.issubset(actual_visited_spots_indices):
            return False
        return True

    def _generate_initial_solution(
        self,
        spots: List[TouristSpot],
        start_spot_index: int,
        distance_matrix: List[List[float]],
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        max_distance: float,
        max_duration: float,
        max_places: int,
        must_visit_spot_indices_set: Set[int],
        transport_speed_kmh: float,
        prev_day_end: Optional[TouristSpot],
        continuity_weight: float,
    ) -> List[int]:
        current_route = [start_spot_index, start_spot_index]
        must_visits_to_insert = list(must_visit_spot_indices_set)
        current_inserted_spots_indices = set()

        if must_visits_to_insert:
            for _ in range(len(must_visits_to_insert)):
                best_must_visit_to_add_idx = -1
                best_route_after_must_visit_insertion = []
                best_objective_for_must_visit = -float("inf")
                insertion_found_this_iteration = False
                remaining_must_visits_to_try = [
                    mv_idx
                    for mv_idx in must_visits_to_insert
                    if mv_idx not in current_inserted_spots_indices
                ]

                if not remaining_must_visits_to_try:
                    break

                for must_visit_idx_to_try in remaining_must_visits_to_try:
                    for i in range(len(current_route) - 1):
                        temp_route = (
                            current_route[: i + 1]
                            + [must_visit_idx_to_try]
                            + current_route[i + 1 :]
                        )
                        temp_must_visit_set_check = (
                            current_inserted_spots_indices.union(
                                {must_visit_idx_to_try}
                            )
                        )
                        obj_val, dist, dur = self._calculate_route_properties(
                            temp_route,
                            spots,
                            distance_matrix,
                            base_scores,
                            priority_scores,
                            transport_speed_kmh,
                            prev_day_end,
                            continuity_weight,
                        )
                        if self._is_route_feasible(
                            temp_route,
                            dist,
                            dur,
                            max_distance,
                            max_duration,
                            max_places,
                            temp_must_visit_set_check,
                            start_spot_index,
                        ):
                            if obj_val > best_objective_for_must_visit:
                                best_objective_for_must_visit = obj_val
                                best_must_visit_to_add_idx = must_visit_idx_to_try
                                best_route_after_must_visit_insertion = temp_route
                                insertion_found_this_iteration = True

                if insertion_found_this_iteration:
                    current_route = best_route_after_must_visit_insertion
                    current_inserted_spots_indices.add(best_must_visit_to_add_idx)
                else:
                    logger.warning(
                        f"초기 해 생성: 필수 방문지 삽입 불가. 현재 경로: {current_route}, 남은 필수 방문지 인덱스: {remaining_must_visits_to_try}"
                    )
                    return [start_spot_index, start_spot_index]

        final_obj_mv, final_dist_mv, final_dur_mv = self._calculate_route_properties(
            current_route,
            spots,
            distance_matrix,
            base_scores,
            priority_scores,
            transport_speed_kmh,
            prev_day_end,
            continuity_weight,
        )
        if not self._is_route_feasible(
            current_route,
            final_dist_mv,
            final_dur_mv,
            max_distance,
            max_duration,
            max_places,
            must_visit_spot_indices_set,
            start_spot_index,
        ):
            logger.warning(
                f"초기 해 생성: 필수 방문지 삽입 후 경로 제약 조건 위반. 필수 인덱스: {must_visit_spot_indices_set}, 경로: {current_route}"
            )
            return [start_spot_index, start_spot_index]

        optional_indices_to_try = list(
            set(range(len(spots))) - {start_spot_index} - must_visit_spot_indices_set
        )
        random.shuffle(optional_indices_to_try)

        for _ in range(len(optional_indices_to_try)):
            if len(set(current_route[1:-1])) >= max_places:
                break
            best_route_after_optional_insertion = []
            current_route_obj_val = self._calculate_route_properties(
                current_route,
                spots,
                distance_matrix,
                base_scores,
                priority_scores,
                transport_speed_kmh,
                prev_day_end,
                continuity_weight,
            )[0]
            best_objective_for_optional = current_route_obj_val
            insertion_found_this_iteration_optional = False
            current_spots_in_route_indices = set(current_route[1:-1])
            remaining_optional_to_try = [
                opt_idx
                for opt_idx in optional_indices_to_try
                if opt_idx not in current_spots_in_route_indices
            ]

            if not remaining_optional_to_try:
                break

            for optional_idx_to_try in remaining_optional_to_try:
                if len(set(current_route[1:-1] + [optional_idx_to_try])) > max_places:
                    continue
                for i in range(len(current_route) - 1):
                    temp_route = (
                        current_route[: i + 1]
                        + [optional_idx_to_try]
                        + current_route[i + 1 :]
                    )
                    obj_val, dist, dur = self._calculate_route_properties(
                        temp_route,
                        spots,
                        distance_matrix,
                        base_scores,
                        priority_scores,
                        transport_speed_kmh,
                        prev_day_end,
                        continuity_weight,
                    )
                    if self._is_route_feasible(
                        temp_route,
                        dist,
                        dur,
                        max_distance,
                        max_duration,
                        max_places,
                        must_visit_spot_indices_set,
                        start_spot_index,
                    ):
                        if obj_val > best_objective_for_optional:
                            best_objective_for_optional = obj_val
                            best_route_after_optional_insertion = temp_route
                            insertion_found_this_iteration_optional = True

            if insertion_found_this_iteration_optional:
                current_route = best_route_after_optional_insertion
            else:
                break
        return current_route

    def _tabu_search_core(
        self,
        initial_route: List[int],
        spots: List[TouristSpot],
        start_spot_index: int,
        distance_matrix: List[List[float]],
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        max_distance: float,
        max_duration: float,
        max_places: int,
        must_visit_spot_indices_set: Set[int],
        transport_speed_kmh: float,
        prev_day_end: Optional[TouristSpot],
        continuity_weight: float,
        max_iterations: int = 1000,
        h_min: int = 10,
        h_max: int = 20,
        t_min: int = 3,
        t_max: int = 7,
    ) -> List[int]:
        logger.info(
            f"TS Core: 초기 경로: {initial_route}, 필수 방문지 인덱스: {must_visit_spot_indices_set}"
        )
        best_solution_route = list(initial_route)
        current_solution_route = list(initial_route)
        best_obj_val, best_dist, best_dur = self._calculate_route_properties(
            best_solution_route,
            spots,
            distance_matrix,
            base_scores,
            priority_scores,
            transport_speed_kmh,
            prev_day_end,
            continuity_weight,
        )

        logger.info(
            f"TS Core: 초기 최적 목적 함수 값: {best_obj_val:.2f}, 거리: {best_dist:.2f}, 시간: {best_dur:.2f}"
        )

        if best_obj_val == -float("inf") or not self._is_route_feasible(
            best_solution_route,
            best_dist,
            best_dur,
            max_distance,
            max_duration,
            max_places,
            must_visit_spot_indices_set,
            start_spot_index,
        ):
            logger.warning(
                "TS Core: 초기 해가 제약 조건을 만족하지 못하여 Tabu Search를 진행할 수 없습니다."
            )
            return initial_route  # Return the (possibly infeasible) initial route

        tabu_list_moves: Dict[Tuple[str, Any], int] = {}

        for iter_count in range(max_iterations):
            best_neighbor_route_candidate = None
            best_neighbor_obj_val_candidate = -float("inf")
            best_move_details_candidate = None
            temp_current_route = list(current_solution_route)
            if (
                len(temp_current_route) > 3
            ):  # Need at least two spots to swap besides start/end
                # Attempt SWAP move
                # Ensure we are swapping actual visited spots, not start/end points if they are part of the list of visited spots
                visitable_indices_in_route = [
                    idx
                    for idx in range(1, len(temp_current_route) - 1)
                    if temp_current_route[idx] != start_spot_index
                ]
                if len(visitable_indices_in_route) >= 2:
                    idx1_route_pos, idx2_route_pos = random.sample(
                        visitable_indices_in_route, 2
                    )
                    node_idx_at_pos1 = temp_current_route[idx1_route_pos]
                    node_idx_at_pos2 = temp_current_route[idx2_route_pos]

                    swapped_route = list(temp_current_route)
                    swapped_route[idx1_route_pos], swapped_route[idx2_route_pos] = (
                        node_idx_at_pos2,
                        node_idx_at_pos1,
                    )
                    move_made_type = "swap"
                    move_attribute = frozenset({node_idx_at_pos1, node_idx_at_pos2})

                    neigh_obj, neigh_dist, neigh_dur = self._calculate_route_properties(
                        swapped_route,
                        spots,
                        distance_matrix,
                        base_scores,
                        priority_scores,
                        transport_speed_kmh,
                        prev_day_end,
                        continuity_weight,
                    )
                    is_feasible = self._is_route_feasible(
                        swapped_route,
                        neigh_dist,
                        neigh_dur,
                        max_distance,
                        max_duration,
                        max_places,
                        must_visit_spot_indices_set,
                        start_spot_index,
                    )

                    if is_feasible:
                        is_tabu = (
                            tabu_list_moves.get((move_made_type, move_attribute), 0)
                            > iter_count
                        )
                        aspiration_met = neigh_obj > best_obj_val
                        if (not is_tabu) or aspiration_met:
                            if neigh_obj > best_neighbor_obj_val_candidate:
                                best_neighbor_obj_val_candidate = neigh_obj
                                best_neighbor_route_candidate = swapped_route
                                best_move_details_candidate = (
                                    move_made_type,
                                    move_attribute,
                                )

            if best_neighbor_route_candidate is not None:
                current_solution_route = list(best_neighbor_route_candidate)
                current_obj_val = best_neighbor_obj_val_candidate
                if best_move_details_candidate:
                    move_type, move_attr = best_move_details_candidate
                    tenure = random.randint(t_min, t_max)
                    tabu_list_moves[(move_type, move_attr)] = iter_count + tenure

                if current_obj_val > best_obj_val:
                    best_solution_route = list(current_solution_route)
                    best_obj_val = current_obj_val
                    # Update best_dist and best_dur as well
                    _, best_dist, best_dur = self._calculate_route_properties(
                        best_solution_route,
                        spots,
                        distance_matrix,
                        base_scores,
                        priority_scores,
                        transport_speed_kmh,
                        prev_day_end,
                        continuity_weight,
                    )
                    logger.debug(
                        f"Iter {iter_count}: New best solution found. Obj: {best_obj_val:.2f}, Route: {best_solution_route}"
                    )

            # Clean up expired tabu moves to prevent memory bloat
            if iter_count % 50 == 0:  # Periodically clean
                tabu_list_moves = {
                    move: expiry
                    for move, expiry in tabu_list_moves.items()
                    if expiry > iter_count
                }

        logger.info(
            f"TS Core: 최종 최적 목적 함수 값: {best_obj_val:.2f}, 거리: {best_dist:.2f}, 시간: {best_dur:.2f}, 경로: {best_solution_route}"
        )
        return best_solution_route

    def optimize_route(
        self,
        daily_spots_input: Union[List[TouristSpot], List[List[TouristSpot]]],
        user_profile: UserProfile,
        daily_start_indices_input: Union[int, List[int]],
        daily_max_distance_input: Union[float, List[float]],
        daily_max_duration_input: Union[float, List[float]],
        max_places_per_day: int,
        base_scores: Dict[int, float],
        priority_scores: Dict[int, Dict[int, float]],
        must_visit_spot_ids_input: Optional[
            Union[Set[int], List[Set[int]]]
        ] = None,  # Added
        ts_max_iterations_per_day: int = 200,
        continuity_weight: float = DEFAULT_CONTINUITY_WEIGHT,
    ) -> List[Tuple[List[int], float, float]]:
        """
        Optimizes routes for single or multiple days using Tabu Search.
        Returns a list of tuples, where each tuple contains (route_indices, total_distance, total_duration).
        """
        is_multi_day = (
            isinstance(daily_spots_input, list)
            and bool(daily_spots_input)
            and isinstance(daily_spots_input[0], list)
        )

        transport_speed_kmh = self._get_transport_speed(user_profile)
        all_optimized_routes_info: List[Tuple[List[int], float, float]] = []
        prev_day_end_spot: Optional[TouristSpot] = None

        daily_spots_list: List[List[TouristSpot]] = (
            daily_spots_input if is_multi_day else [daily_spots_input]
        )  # type: ignore
        daily_start_indices_list: List[int] = (
            daily_start_indices_input if is_multi_day else [daily_start_indices_input]
        )  # type: ignore
        daily_max_distance_list: List[float] = (
            daily_max_distance_input if is_multi_day else [daily_max_distance_input]
        )  # type: ignore
        daily_max_duration_list: List[float] = (
            daily_max_duration_input if is_multi_day else [daily_max_duration_input]
        )  # type: ignore

        # Normalize must_visit_spot_ids_input
        daily_must_visit_spot_ids_list: List[Optional[Set[int]]]
        if must_visit_spot_ids_input is None:
            daily_must_visit_spot_ids_list = [None] * len(daily_spots_list)
        elif isinstance(must_visit_spot_ids_input, set):
            daily_must_visit_spot_ids_list = (
                [must_visit_spot_ids_input]
                if not is_multi_day
                else [must_visit_spot_ids_input] + [None] * (len(daily_spots_list) - 1)
            )  # Apply to first day if multi-day and single set given
        elif isinstance(must_visit_spot_ids_input, list):
            if len(must_visit_spot_ids_input) == len(daily_spots_list):
                daily_must_visit_spot_ids_list = must_visit_spot_ids_input
            else:
                logger.error(
                    "Length mismatch: must_visit_spot_ids_input list and daily_spots_list"
                )
                return []  # Or raise error
        else:
            logger.error(
                f"Invalid type for must_visit_spot_ids_input: {type(must_visit_spot_ids_input)}"
            )
            return []

        for day_idx, current_day_spots in enumerate(daily_spots_list):
            if not current_day_spots:
                logger.warning(f"Day {day_idx + 1}: No spots provided. Skipping.")
                all_optimized_routes_info.append(([], 0.0, 0.0))
                prev_day_end_spot = None  # Reset for next day if this day is empty
                continue

            start_spot_index_for_day = daily_start_indices_list[day_idx]
            max_dist_for_day = daily_max_distance_list[day_idx]
            max_dur_for_day = daily_max_duration_list[day_idx]
            must_visit_ids_for_day: Optional[Set[int]] = daily_must_visit_spot_ids_list[
                day_idx
            ]

            # Convert must_visit_ids (spot IDs) to must_visit_indices (indices in current_day_spots)
            must_visit_spot_indices_set_for_day: Set[int] = set()
            spot_id_to_idx_map = {
                self._get_spot_id(spot): i for i, spot in enumerate(current_day_spots)
            }
            if must_visit_ids_for_day:
                for spot_id in must_visit_ids_for_day:
                    if spot_id in spot_id_to_idx_map:
                        must_visit_spot_indices_set_for_day.add(
                            spot_id_to_idx_map[spot_id]
                        )
                    else:
                        logger.warning(
                            f"Day {day_idx + 1}: Must-visit spot ID {spot_id} not found in current day's spots. Ignoring."
                        )

            # Ensure start_spot_index is valid for current_day_spots
            if not (0 <= start_spot_index_for_day < len(current_day_spots)):
                logger.error(
                    f"Day {day_idx + 1}: Invalid start_spot_index {start_spot_index_for_day} for {len(current_day_spots)} spots. Skipping day."
                )
                all_optimized_routes_info.append(([], 0.0, 0.0))
                prev_day_end_spot = None
                continue

            distance_matrix_for_day = self.calculate_distance_matrix(current_day_spots)

            initial_route_indices = self._generate_initial_solution(
                spots=current_day_spots,
                start_spot_index=start_spot_index_for_day,
                distance_matrix=distance_matrix_for_day,
                base_scores=base_scores,
                priority_scores=priority_scores,
                max_distance=max_dist_for_day,
                max_duration=max_dur_for_day,
                max_places=max_places_per_day,
                must_visit_spot_indices_set=must_visit_spot_indices_set_for_day,
                transport_speed_kmh=transport_speed_kmh,
                prev_day_end=prev_day_end_spot,
                continuity_weight=continuity_weight,
            )

            if (
                not initial_route_indices or len(initial_route_indices) < 2
            ):  # Check if initial solution is valid
                logger.warning(
                    f"Day {day_idx + 1}: Failed to generate a valid initial solution. Route: {initial_route_indices}"
                )
                # Check if it's just start-end, meaning no feasible route found for must-visits
                if initial_route_indices == [
                    start_spot_index_for_day,
                    start_spot_index_for_day,
                ]:
                    all_optimized_routes_info.append((initial_route_indices, 0.0, 0.0))
                else:  # Truly empty or invalid
                    all_optimized_routes_info.append(([], 0.0, 0.0))
                prev_day_end_spot = None  # Reset continuity if this day fails
                continue

            optimized_route_indices_for_day = self._tabu_search_core(
                initial_route=initial_route_indices,
                spots=current_day_spots,
                start_spot_index=start_spot_index_for_day,
                distance_matrix=distance_matrix_for_day,
                base_scores=base_scores,
                priority_scores=priority_scores,
                max_distance=max_dist_for_day,
                max_duration=max_dur_for_day,
                max_places=max_places_per_day,
                must_visit_spot_indices_set=must_visit_spot_indices_set_for_day,
                transport_speed_kmh=transport_speed_kmh,
                prev_day_end=prev_day_end_spot,
                continuity_weight=continuity_weight,
                max_iterations=ts_max_iterations_per_day,
            )

            final_score, final_dist, final_dur = self._calculate_route_properties(
                optimized_route_indices_for_day,
                current_day_spots,
                distance_matrix_for_day,
                base_scores,
                priority_scores,
                transport_speed_kmh,
                prev_day_end_spot,
                continuity_weight,
            )

            # Validate final route from Tabu Search
            if not self._is_route_feasible(
                optimized_route_indices_for_day,
                final_dist,
                final_dur,
                max_dist_for_day,
                max_dur_for_day,
                max_places_per_day,
                must_visit_spot_indices_set_for_day,
                start_spot_index_for_day,
            ):
                logger.warning(
                    f"Day {day_idx + 1}: Tabu search resulted in an infeasible route. Route: {optimized_route_indices_for_day}, Dist: {final_dist}, Dur: {final_dur}. Falling back to initial or empty."
                )
                # Fallback to initial solution if it was feasible, otherwise empty
                initial_score, initial_dist, initial_dur = (
                    self._calculate_route_properties(
                        initial_route_indices,
                        current_day_spots,
                        distance_matrix_for_day,
                        base_scores,
                        priority_scores,
                        transport_speed_kmh,
                        prev_day_end_spot,
                        continuity_weight,
                    )
                )
                if self._is_route_feasible(
                    initial_route_indices,
                    initial_dist,
                    initial_dur,
                    max_dist_for_day,
                    max_dur_for_day,
                    max_places_per_day,
                    must_visit_spot_indices_set_for_day,
                    start_spot_index_for_day,
                ):
                    logger.info(
                        f"Day {day_idx + 1}: Falling back to initial feasible solution."
                    )
                    all_optimized_routes_info.append(
                        (initial_route_indices, initial_dist, initial_dur)
                    )
                    if (
                        initial_route_indices
                        and len(initial_route_indices) > 1
                        and initial_route_indices[-2] < len(current_day_spots)
                    ):
                        prev_day_end_spot = current_day_spots[
                            initial_route_indices[-2]
                        ]  # Second to last is the actual last visited spot
                    else:
                        prev_day_end_spot = None
                else:
                    logger.info(
                        f"Day {day_idx + 1}: No feasible solution found, returning empty route for the day."
                    )
                    all_optimized_routes_info.append(([], 0.0, 0.0))
                    prev_day_end_spot = None
            else:
                all_optimized_routes_info.append(
                    (optimized_route_indices_for_day, final_dist, final_dur)
                )
                if (
                    optimized_route_indices_for_day
                    and len(optimized_route_indices_for_day) > 1
                    and optimized_route_indices_for_day[-2] < len(current_day_spots)
                ):
                    prev_day_end_spot = current_day_spots[
                        optimized_route_indices_for_day[-2]
                    ]  # Second to last is the actual last visited spot
                else:
                    prev_day_end_spot = None

        return all_optimized_routes_info
