"""
클러스터 점수화 모델 모듈
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.value_objects.coordinate import Coordinate


class ClusterScoringModel:
    """클러스터 점수화를 위한 머신러닝 모델"""

    def __init__(self, model_path: Optional[str] = None):
        """
        초기화

        Args:
            model_path: 모델 파일 경로 (선택 사항)
        """
        self.features_scaler = StandardScaler()
        self.scoring_model = RandomForestRegressor(max_depth=10, random_state=42)

        # 모델 파일이 제공된 경우 로드
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def extract_cluster_features(
        self,
        spots: List[TouristSpot],
        user_profile: UserProfile,
        center_coord: Coordinate,
        day_index: int,
        num_days: int,
    ) -> np.ndarray:
        """
        클러스터 특성 추출

        Args:
            spots: 클러스터 내 관광지 목록
            user_profile: 사용자 프로필
            center_coord: 클러스터 중심 좌표
            day_index: 일차 인덱스 (0부터 시작)
            num_days: 총 일수

        Returns:
            특성 벡터
        """
        features = [len(spots)]

        # 클러스터 크기

        # 사용자 테마와 일치하는 관광지 비율
        if user_profile.themes:
            theme_match_count = sum(
                1
                for spot in spots
                if any(theme in spot.category for theme in user_profile.themes)
            )
            theme_match_ratio = theme_match_count / len(spots) if spots else 0
            features.append(theme_match_ratio)
        else:
            features.append(0.0)

        # 반드시 방문해야 하는 관광지 수
        must_visit_count = sum(
            1 for spot in spots if spot.tourist_spot_id in user_profile.must_visit_list
        )
        features.append(must_visit_count)

        # 방문하지 않을 관광지 수
        not_visit_count = sum(
            1 for spot in spots if spot.tourist_spot_id in user_profile.not_visit_list
        )
        features.append(not_visit_count)

        # 평균 활동 수준
        avg_activity_level = (
            np.mean([spot.activity_level for spot in spots]) if spots else 0
        )
        features.append(avg_activity_level)

        # 클러스터 밀도 (관광지 간 평균 거리의 역수)
        if len(spots) > 1:
            distances = []
            for i in range(len(spots)):
                for j in range(i + 1, len(spots)):
                    coord_i = Coordinate(
                        spots[i].coordinate.latitude, spots[i].coordinate.longitude
                    )
                    coord_j = Coordinate(
                        spots[j].coordinate.latitude, spots[j].coordinate.longitude
                    )
                    distances.append(coord_i.distance_to(coord_j))
            avg_distance = np.mean(distances) if distances else 0
            density = 1.0 / (avg_distance + 0.1)  # 0으로 나누기 방지
            features.append(density)
        else:
            features.append(0.0)

        # 일차 정보 (정규화)
        features.append(day_index / num_days)

        # 이전 일차 클러스터와의 거리 (첫째 날은 0)
        features.append(0.0)  # 실제 구현에서는 이전 클러스터 중심과의 거리 계산

        return np.array(features).reshape(1, -1)

    def train(self, cluster_data: List[Dict[str, Any]], ratings: List[float]) -> None:
        """
        모델 학습

        Args:
            cluster_data: 클러스터 데이터 목록
            ratings: 클러스터 평점 목록
        """
        if not cluster_data or not ratings or len(cluster_data) != len(ratings):
            return

        # 특성 추출
        X = []
        for data in cluster_data:
            features = self.extract_cluster_features(
                spots=data["spots"],
                user_profile=data["user_profile"],
                center_coord=data["center_coord"],
                day_index=data["day_index"],
                num_days=data["num_days"],
            )
            X.append(features.flatten())

        X = np.array(X)
        y = np.array(ratings)

        # 특성 스케일링
        X_scaled = self.features_scaler.fit_transform(X)

        # 모델 학습
        self.scoring_model.fit(X_scaled, y)

    def score_cluster(
        self,
        spots: List[TouristSpot],
        user_profile: UserProfile,
        center_coord: Coordinate,
        day_index: int,
        num_days: int,
    ) -> float:
        """
        클러스터 점수 계산

        Args:
            spots: 클러스터 내 관광지 목록
            user_profile: 사용자 프로필
            center_coord: 클러스터 중심 좌표
            day_index: 일차 인덱스 (0부터 시작)
            num_days: 총 일수

        Returns:
            클러스터 점수 (0.0 ~ 10.0)
        """
        # 특성 추출
        features = self.extract_cluster_features(
            spots=spots,
            user_profile=user_profile,
            center_coord=center_coord,
            day_index=day_index,
            num_days=num_days,
        )

        # 특성 스케일링
        features_scaled = self.features_scaler.transform(features)

        # 점수 예측
        score = self.scoring_model.predict(features_scaled)[0]

        # 0.0 ~ 10.0 범위로 클리핑
        return max(0.0, min(10.0, score))

    def rank_clusters(
        self,
        clusters: Dict[int, List[TouristSpot]],
        user_profile: UserProfile,
        num_days: int,
        base_scores: Dict[int, float],
    ) -> List[Tuple[int, float]]:
        """
        클러스터 순위 계산

        Args:
            clusters: 클러스터 ID를 키로, 관광지 목록을 값으로 하는 딕셔너리
            user_profile: 사용자 프로필
            num_days: 총 일수
            base_scores: 각 관광지의 기본 선호도 점수 (spot.id -> score)

        Returns:
            (클러스터 ID, 점수) 튜플 목록 (점수 내림차순 정렬)
        """

        cluster_scores = []

        for cluster_id, spots in clusters.items():
            if not spots:
                continue

            # 클러스터 중심 좌표 계산
            latitudes = [spot.coordinate.latitude for spot in spots]
            longitudes = [spot.coordinate.longitude for spot in spots]
            center_coord = Coordinate(
                latitude=float(np.mean(latitudes)), longitude=float(np.mean(longitudes))
            )

            # 만약 features_scaler가 아직 fit되지 않았다면 ML 모델 대신 base_scores 평균 사용
            if hasattr(self.features_scaler, "mean_"):
                score_ml = self.score_cluster(
                    spots=spots,
                    user_profile=user_profile,
                    center_coord=center_coord,
                    day_index=0,
                    num_days=num_days,
                )
            else:
                base_list = [base_scores.get(spot.tourist_spot_id, 0) for spot in spots]
                score_ml = np.mean(base_list) if base_list else 0.0

            # base_scores에 따른 클러스터의 평균 기본 점수 계산
            base_list = [
                base_scores.get(spot.tourist_spot_id, score_ml) for spot in spots
            ]
            avg_base_score = np.mean(base_list) if base_list else score_ml

            # 두 점수를 평균하여 최종 클러스터 점수 산출
            combined_score = (score_ml + avg_base_score) / 2.0

            cluster_scores.append((cluster_id, combined_score))

        # 점수 내림차순 정렬 후 반환
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        return cluster_scores

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
        # base_scores 기본값 지정 (여기서는 빈 dict 사용)
        default_base_scores: Dict[int, float] = {}
        # 클러스터 점수 계산 (base_scores 추가)
        cluster_scores = self.rank_clusters(
            clusters, user_profile, num_days, default_base_scores
        )

        # 상위 N개 클러스터 선택
        top_clusters = [cluster_id for cluster_id, _ in cluster_scores[:num_days]]

        # 클러스터 중심 좌표 계산
        cluster_centers = {}
        for cluster_id in top_clusters:
            spots = clusters[cluster_id]
            latitudes = [spot.coordinate.latitude for spot in spots]
            longitudes = [spot.coordinate.longitude for spot in spots]
            cluster_centers[cluster_id] = Coordinate(
                latitude=float(np.mean(latitudes)), longitude=float(np.mean(longitudes))
            )

        # 클러스터 간 거리 계산
        distances = {}
        for i, cluster_id1 in enumerate(top_clusters):
            for j, cluster_id2 in enumerate(top_clusters):
                if i != j:
                    center1 = cluster_centers[cluster_id1]
                    center2 = cluster_centers[cluster_id2]
                    distances[(cluster_id1, cluster_id2)] = center1.distance_to(center2)

        # 최적 순서 계산 (그리디 알고리즘)
        # 첫 번째 클러스터는 점수가 가장 높은 클러스터
        sequence = [top_clusters[0]]
        remaining = set(top_clusters[1:])

        # 나머지 클러스터 추가
        while remaining:
            last_cluster = sequence[-1]
            # 가장 가까운 클러스터 찾기
            next_cluster = min(
                remaining,
                key=lambda cluster_id: distances.get(
                    (last_cluster, cluster_id), float("inf")
                ),
            )
            sequence.append(next_cluster)
            remaining.remove(next_cluster)

        return sequence

    def save_model(self, model_path: str) -> None:
        """
        모델 저장

        Args:
            model_path: 모델 파일 경로
        """
        model_data = {
            "features_scaler": self.features_scaler,
            "scoring_model": self.scoring_model,
        }
        joblib.dump(model_data, model_path)

    def load_model(self, model_path: str) -> None:
        """
        모델 로드

        Args:
            model_path: 모델 파일 경로
        """
        model_data = joblib.load(model_path)
        self.features_scaler = model_data["features_scaler"]
        self.scoring_model = model_data["scoring_model"]
