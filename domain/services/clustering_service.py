from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from domain.entities.tourist_spot import TouristSpot
from domain.value_objects.coordinate import Coordinate


class ClusteringService:
    @staticmethod
    def cluster_tourist_spots(
        spots: List[TouristSpot], num_clusters: int, random_state: int = 42
    ) -> Dict[int, List[TouristSpot]]:
        """
        관광지를 좌표 기반으로 KMeans 클러스터링

        Args:
            spots: 클러스터링할 관광지 목록
            num_clusters: 클러스터 수
            random_state: 랜덤 시드

        Returns:
            클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
        """
        if not spots or num_clusters <= 0 or num_clusters > len(spots):
            return {}

        coordinates = np.array(
            [[spot.coordinate.latitude, spot.coordinate.longitude] for spot in spots]
        )

        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(coordinates)

        clusters: Dict[int, List[TouristSpot]] = {}
        for spot, label in zip(spots, labels):
            clusters.setdefault(label, []).append(spot)

        return clusters

    @staticmethod
    def calculate_cluster_center(spots: List[TouristSpot]):
        """
        관광지 목록의 중심 좌표 계산

        Args:
            spots: 관광지 목록

        Returns:
            중심 좌표
        """
        if not spots:
            raise ValueError("관광지 목록이 비어있습니다.")

        total_lat = sum(spot.coordinate.latitude for spot in spots)
        total_lon = sum(spot.coordinate.longitude for spot in spots)
        avg_lat = total_lat / len(spots)
        avg_lon = total_lon / len(spots)

        return Coordinate(latitude=avg_lat, longitude=avg_lon)

    @staticmethod
    def calculate_cluster_centers(
        clusters: Dict[int, List[TouristSpot]],
    ) -> Dict[int, Coordinate]:
        """
        클러스터별 중심 좌표 계산

        Args:
            clusters: 클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리

        Returns:
            클러스터 ID를 키로, 해당 클러스터의 중심 좌표를 값으로 하는 딕셔너리
        """
        centers = {}
        for cluster_id, spots in clusters.items():
            if spots:
                centers[cluster_id] = ClusteringService.calculate_cluster_center(spots)

        return centers

    @staticmethod
    def calculate_cluster_radius(spots: List[TouristSpot], center: Coordinate):
        """
        클러스터의 반경 계산 (중심에서 가장 먼 관광지까지의 거리)

        Args:
            spots: 관광지 목록
            center: 클러스터 중심 좌표

        Returns:
            클러스터 반경 (km)
        """
        if not spots:
            return 0.0

        distances = [
            center.distance_to(
                Coordinate(spot.coordinate.latitude, spot.coordinate.longitude)
            )
            for spot in spots
        ]
        return max(distances) if distances else 0.0

    @staticmethod
    def calculate_cluster_density(spots: List[TouristSpot], radius_km: float) -> float:
        """
        클러스터의 밀도 계산 (단위 면적당 관광지 수)

        Args:
            spots: 관광지 목록
            radius_km: 클러스터 반경 (km)

        Returns:
            클러스터 밀도 (관광지 수 / km²)
        """
        if not spots or radius_km <= 0:
            return 0.0

        area = np.pi * (radius_km**2)
        return len(spots) / area

    @staticmethod
    def calculate_inter_cluster_distances(
        clusters: Dict[int, List[TouristSpot]],
    ) -> Dict[Tuple[int, int], float]:
        """
        클러스터 간 거리 계산

        Args:
            clusters: 클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리

        Returns:
            클러스터 ID 쌍을 키로, 거리를 값으로 하는 딕셔너리
        """
        distances = {}
        cluster_centers = {}

        for cluster_id, spots in clusters.items():
            if spots:
                cluster_centers[cluster_id] = (
                    ClusteringService.calculate_cluster_center(spots)
                )

        cluster_ids = list(clusters.keys())
        for i, id1 in enumerate(cluster_ids):
            for id2 in cluster_ids[i + 1 :]:
                if id1 in cluster_centers and id2 in cluster_centers:
                    distance = cluster_centers[id1].distance_to(cluster_centers[id2])
                    distances[(id1, id2)] = distance
                    distances[(id2, id1)] = distance

        return distances

    @staticmethod
    def cluster_spots_kmeans(
        spots: List[TouristSpot], num_clusters: int, random_state: int = 42
    ) -> Dict[int, List[TouristSpot]]:
        """
        KMeans 기반 관광지 클러스터링

        테스트에서 호출하는 cluster_spots_kmeans는 기존 cluster_tourist_spots와 동일한 기능을 수행합니다.
        """
        return ClusteringService.cluster_tourist_spots(
            spots, num_clusters, random_state
        )

    @staticmethod
    def cluster_spots_dbscan(
        spots: List[TouristSpot], eps_km: float, min_samples: int
    ) -> Dict[int, List[TouristSpot]]:
        """
        DBSCAN 기반 관광지 클러스터링

        Args:
            spots: 클러스터링할 관광지 목록
            eps_km: 클러스터링 반경 (킬로미터 단위)
            min_samples: 최소 샘플 수

        Returns:
            클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
            (노이즈로 판단된 관광지는 -1번 클러스터에 포함됩니다.)
        """
        if not spots:
            return {}

        coordinates = np.radians(
            np.array(
                [
                    [spot.coordinate.latitude, spot.coordinate.longitude]
                    for spot in spots
                ]
            )
        )
        eps_radians = eps_km / 6371.0

        dbscan = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
        labels = dbscan.fit_predict(coordinates)

        clusters: Dict[int, List[TouristSpot]] = {}
        for spot, label in zip(spots, labels):
            clusters.setdefault(label, []).append(spot)

        return clusters

    @staticmethod
    def extract_coordinates(spots: List[TouristSpot]) -> np.ndarray:
        """
        관광지 목록으로부터 좌표 배열 추출

        Args:
            spots: 관광지 목록

        Returns:
            numpy array 형태의 좌표 (각 행: [latitude, longitude])
        """
        if not spots:
            return np.array([])
        return np.array(
            [[spot.coordinate.latitude, spot.coordinate.longitude] for spot in spots]
        )

    @staticmethod
    def find_optimal_clusters(
        spots: List[TouristSpot],
        min_clusters: int,
        max_clusters: int,
        random_state: int = 42,
    ) -> int:
        """
        관광지 데이터를 대상으로 최적의 클러스터 수를 찾음 (silhouette score 기반)

        Args:
            spots: 관광지 목록
            min_clusters: 최소 클러스터 수 (보통 2 이상)
            max_clusters: 최대 클러스터 수
            random_state: 랜덤 시드

        Returns:
            silhouette score가 가장 높은 클러스터 수
        """

        if (
            not spots
            or min_clusters < 2
            or max_clusters > len(spots)
            or min_clusters > max_clusters
        ):
            raise ValueError("유효하지 않은 클러스터 수 범위입니다.")

        coordinates = np.array(
            [[spot.coordinate.latitude, spot.coordinate.longitude] for spot in spots]
        )
        best_k = min_clusters
        best_score = -1

        for k in range(min_clusters, max_clusters + 1):
            if k == 1:
                continue
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(coordinates)
            score = silhouette_score(coordinates, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k
