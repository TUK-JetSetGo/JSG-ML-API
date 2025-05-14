"""
개선된 클러스터링 알고리즘 모듈
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from infrastructure.adapters.ml.cluster_scoring_model import \
    ClusterScoringModel


class EnhancedClusteringAlgorithm:
    """개선된 클러스터링 알고리즘"""

    def __init__(self, cluster_scoring_model: Optional[ClusterScoringModel] = None):
        """
        초기화

        Args:
            cluster_scoring_model: 클러스터 점수화 모델 (선택 사항)
        """
        self.cluster_scoring_model = cluster_scoring_model or ClusterScoringModel()

    def find_optimal_clusters(
        self, spots: List[TouristSpot], min_clusters: int = 2, max_clusters: int = 10
    ) -> int:
        """
        최적의 클러스터 수 찾기 (실루엣 점수 기반)

        Args:
            spots: 관광지 목록
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수

        Returns:
            최적의 클러스터 수
        """
        if len(spots) < min_clusters:
            return 1

        # 좌표 데이터 추출
        coordinates = np.array(
            [[spot.coordinate.latitude, spot.coordinate.longitude] for spot in spots]
        )

        # 실루엣 점수 계산
        silhouette_scores = []
        for n_clusters in range(min_clusters, min(max_clusters + 1, len(spots))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)

            # 클러스터가 하나만 있는 경우 실루엣 점수를 계산할 수 없음
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(coordinates, cluster_labels)
                silhouette_scores.append((n_clusters, score))

        if not silhouette_scores:
            return min_clusters

        # 최적의 클러스터 수 선택
        optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]

        return optimal_n_clusters

    def select_optimal_clusters(
        self,
        clusters: Dict[int, List[TouristSpot]],
        user_profile: UserProfile,
        num_days: int,
        base_scores: Dict[int, float],
    ) -> List[List[TouristSpot]]:
        """
        선택된 클러스터를 반환합니다.

        Args:
            clusters: 클러스터링 결과 (클러스터 ID -> 관광지 리스트)
            user_profile: 사용자 프로필
            num_days: 여행 일수
            base_scores: 각 관광지의 기본 선호도 점수 (spot.id -> score)

        Returns:
            일자별 할당된 클러스터 리스트 (예: [[TouristSpot, ...], [TouristSpot, ...], ...])
        """
        # 클러스터 점수 계산 (base_scores를 반영)
        cluster_scores = self.cluster_scoring_model.rank_clusters(
            clusters, user_profile, num_days, base_scores
        )

        # 반드시 방문해야 하는 관광지가 포함된 클러스터 식별
        must_visit_clusters = set()
        for spot_id in user_profile.must_visit_list:
            for cluster_id, cluster_spots in clusters.items():
                if any(spot.tourist_spot_id == spot_id for spot in cluster_spots):
                    must_visit_clusters.add(cluster_id)

        # 우선 반드시 방문해야 하는 클러스터 선택
        selected_cluster_ids = []
        for cluster_id, score in cluster_scores:
            if cluster_id in must_visit_clusters:
                selected_cluster_ids.append(cluster_id)
                must_visit_clusters.remove(cluster_id)
                if len(selected_cluster_ids) >= num_days:
                    break

        # 남은 클러스터 중 점수가 높은 순서대로 선택
        for cluster_id, score in cluster_scores:
            if cluster_id not in selected_cluster_ids:
                selected_cluster_ids.append(cluster_id)
                if len(selected_cluster_ids) >= num_days:
                    break

        # 선택된 클러스터만 리스트로 반환
        result_clusters = [
            clusters[cluster_id] for cluster_id in selected_cluster_ids[:num_days]
        ]
        return result_clusters

    def cluster_with_kmeans(
        self, spots: List[TouristSpot], num_clusters: int, random_state: int = 42
    ) -> Dict[int, List[TouristSpot]]:
        """
        K-means 클러스터링

        Args:
            spots: 관광지 목록
            num_clusters: 클러스터 수
            random_state: 랜덤 시드

        Returns:
            클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
        """
        if not spots or num_clusters <= 0 or num_clusters > len(spots):
            return {}

        # 좌표 데이터 추출
        coordinates = np.array(
            [[spot.coordinate.latitude, spot.coordinate.longitude] for spot in spots]
        )

        # K-means 클러스터링 수행
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(coordinates)

        # 클러스터별로 관광지 그룹화
        clusters: Dict[int, List[TouristSpot]] = {}
        for spot, label in zip(spots, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(spot)

        return clusters

    def cluster_with_dbscan(
        self, spots: List[TouristSpot], eps_km: float = 5.0, min_samples: int = 3
    ) -> Dict[int, List[TouristSpot]]:
        """
        DBSCAN 클러스터링

        Args:
            spots: 관광지 목록
            eps_km: 이웃 반경 (km)
            min_samples: 핵심 포인트를 정의하기 위한 최소 샘플 수

        Returns:
            클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
        """
        if not spots:
            return {}

        # 좌표 데이터 추출 (라디안 단위로 변환)
        coordinates = np.radians(
            np.array(
                [
                    [spot.coordinate.latitude, spot.coordinate.longitude]
                    for spot in spots
                ]
            )
        )

        # 지구 반경 (km)
        earth_radius = 6371.0

        # eps를 라디안 단위로 변환
        eps_rad = eps_km / earth_radius

        # DBSCAN 클러스터링 수행 (haversine 거리 사용)
        dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
        labels = dbscan.fit_predict(coordinates)

        # 클러스터별로 관광지 그룹화
        clusters: Dict[int, List[TouristSpot]] = {}
        for spot, label in zip(spots, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(spot)

        return clusters

    def cluster_with_user_preferences(
        self,
        spots: List[TouristSpot],
        user_profile: UserProfile,
        num_days: int,
        method: str = "kmeans",
    ) -> Dict[int, List[TouristSpot]]:
        """
        사용자 선호도를 고려한 클러스터링

        Args:
            spots: 관광지 목록
            user_profile: 사용자 프로필
            num_days: 일수
            method: 클러스터링 방법 ('kmeans' 또는 'dbscan')

        Returns:
            클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
        """
        if not spots:
            return {}

        # 방문하지 않을 장소 제외
        filtered_spots = [
            spot
            for spot in spots
            if spot.tourist_spot_id not in user_profile.not_visit_list
        ]

        if not filtered_spots:
            return {}

        # 최적의 클러스터 수 찾기
        if method == "kmeans":
            # 일수보다 많은 클러스터를 생성하여 나중에 병합
            num_clusters = min(
                self.find_optimal_clusters(filtered_spots), len(filtered_spots)
            )
            num_clusters = max(num_clusters, num_days)

            # K-means 클러스터링 수행
            clusters = self.cluster_with_kmeans(filtered_spots, num_clusters)
        else:
            # DBSCAN 클러스터링 수행
            clusters = self.cluster_with_dbscan(filtered_spots)

            # 클러스터 수가 일수보다 적은 경우 K-means로 보완
            if len(clusters) < num_days:
                clusters = self.cluster_with_kmeans(filtered_spots, num_days)

        # 반드시 방문해야 하는 장소가 포함된 클러스터 확인
        must_visit_clusters = set()
        for spot_id in user_profile.must_visit_list:
            for cluster_id, cluster_spots in clusters.items():
                if any(spot.tourist_spot_id == spot_id for spot in cluster_spots):
                    must_visit_clusters.add(cluster_id)

        # 클러스터 점수 계산 및 순위 지정
        cluster_scores = self.cluster_scoring_model.rank_clusters(
            clusters, user_profile, num_days, {}
        )

        # 반드시 방문해야 하는 클러스터를 우선 선택
        selected_clusters = []
        for cluster_id, _ in cluster_scores:
            if cluster_id in must_visit_clusters:
                selected_clusters.append(cluster_id)
                must_visit_clusters.remove(cluster_id)
                if len(selected_clusters) >= num_days:
                    break

        # 나머지 클러스터 선택
        for cluster_id, _ in cluster_scores:
            if cluster_id not in selected_clusters:
                selected_clusters.append(cluster_id)
                if len(selected_clusters) >= num_days:
                    break

        # 선택된 클러스터만 반환
        result_clusters = {
            i: clusters[cluster_id]
            for i, cluster_id in enumerate(selected_clusters[:num_days])
        }

        return result_clusters

    def cluster_spots_for_multi_day_trip(
        self,
        spots: List[TouristSpot],
        user_profile: UserProfile,
        num_days: int,
        base_scores: Dict[int, float],
    ) -> Dict[int, List[TouristSpot]]:
        """
        다중 일자 여행 일정에 맞게 관광지를 클러스터링합니다.

        Args:
            spots: 관광지 목록.
            user_profile: 사용자 프로필.
            num_days: 여행 일수.
            base_scores: 각 관광지의 기본 선호도 점수 (spot.id -> score).

        Returns:
            각 일자별로 할당된 관광지 클러스터 딕셔너리.
        """
        # 사용자 선호에 따라 방문하지 않을 관광지는 제외
        filtered_spots = [
            spot
            for spot in spots
            if spot.tourist_spot_id not in user_profile.not_visit_list
        ]
        if not filtered_spots:
            return {}

        # 최적의 클러스터 수를 결정 (일수보다 적지 않도록 보정)
        num_clusters = max(num_days, self.find_optimal_clusters(filtered_spots))

        # K-means를 사용하여 클러스터링 수행
        clusters = self.cluster_with_kmeans(filtered_spots, num_clusters)

        # 반드시 방문해야 하는 관광지가 포함된 클러스터 식별
        must_visit_clusters = set()
        for spot_id in user_profile.must_visit_list:
            for cluster_id, cluster_spots in clusters.items():
                if any(spot.tourist_spot_id == spot_id for spot in cluster_spots):
                    must_visit_clusters.add(cluster_id)

        # 클러스터 점수 계산: 기존의 클러스터 점수화 모델에 base_scores를 전달합니다.
        cluster_scores = self.cluster_scoring_model.rank_clusters(
            clusters, user_profile, num_days, base_scores
        )

        # 반드시 방문해야 하는 클러스터를 우선 선택하고, 나머지 클러스터 중 높은 점수 순으로 선택
        selected_clusters = []
        for cluster_id, _ in cluster_scores:
            if cluster_id in must_visit_clusters:
                selected_clusters.append(cluster_id)
                must_visit_clusters.remove(cluster_id)
                if len(selected_clusters) >= num_days:
                    break
        for cluster_id, _ in cluster_scores:
            if cluster_id not in selected_clusters:
                selected_clusters.append(cluster_id)
                if len(selected_clusters) >= num_days:
                    break

        # 선택된 클러스터만 일자별로 매핑하여 반환 (0부터 num_days-1)
        result_clusters = {
            i: clusters[cluster_id]
            for i, cluster_id in enumerate(selected_clusters[:num_days])
        }
        return result_clusters

    def optimize_cluster_sequence(
        self,
        clusters: Dict[int, List[TouristSpot]],
        user_profile: UserProfile,
        num_days: int,
    ) -> List[int]:
        """
        최적 클러스터 순서 계산

        Args:
            clusters: 클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
            user_profile: 사용자 프로필
            num_days: 총 일수

        Returns:
            클러스터 ID 목록 (최적 순서)
        """
        return self.cluster_scoring_model.optimize_cluster_sequence(
            clusters, user_profile, num_days
        )

    def visualize_clusters(
        self, clusters: Dict[int, List[TouristSpot]], output_path: str = "clusters.png"
    ) -> None:
        """
        클러스터 시각화

        Args:
            clusters: 클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
            output_path: 출력 파일 경로
        """
        plt.figure(figsize=(10, 8))

        # 색상 목록
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "brown",
        ]

        # 각 클러스터 시각화
        for cluster_id, spots in clusters.items():
            color = colors[cluster_id % len(colors)]

            # 관광지 좌표 추출
            latitudes = [spot.coordinate.latitude for spot in spots]
            longitudes = [spot.coordinate.longitude for spot in spots]

            # 클러스터 중심 계산
            center_lat = np.mean(latitudes)
            center_lon = np.mean(longitudes)

            # 관광지 산점도
            plt.scatter(
                longitudes, latitudes, c=color, label=f"Cluster {cluster_id}", alpha=0.7
            )

            # 클러스터 중심 표시
            plt.scatter(
                center_lon, center_lat, c=color, marker="X", s=100, edgecolors="black"
            )

        plt.title("Tourist Spot Clusters")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)

        # 파일 저장
        plt.savefig(output_path)
        plt.close()
