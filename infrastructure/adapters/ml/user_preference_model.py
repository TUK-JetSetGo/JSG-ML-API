"""
사용자 선호도 예측 모델 모듈
"""

import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from domain.entities.tourist_spot import TouristSpot
from domain.entities.user_profile import UserProfile
from domain.value_objects.category import CategorySet


class UserPreferenceModel:
    """사용자 선호도 예측을 위한 머신러닝 모델"""

    def __init__(self, model_path: Optional[str] = None):
        """
        초기화

        Args:
            model_path: 모델 파일 경로 (선택 사항)
        """
        self.user_features_scaler = StandardScaler()
        self.spot_features_scaler = StandardScaler()
        self.user_pca = PCA(n_components=10)
        self.spot_pca = PCA(n_components=10)
        self.preference_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.user_clusters = KMeans(n_clusters=5, random_state=42)
        self.spot_clusters = KMeans(n_clusters=8, random_state=42)
        self.knn_model = NearestNeighbors(n_neighbors=10, algorithm="auto")

        # 모델 파일이 제공된 경우 로드
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def extract_user_features(self, user_profile: UserProfile) -> np.ndarray:
        """
        사용자 프로필에서 특성 추출

        Args:
            user_profile: 사용자 프로필

        Returns:
            특성 벡터
        """
        # 사용자 프로필에서 특성 벡터 생성 (내부 전처리)
        user_profile.generate_feature_vector()

        features = []

        # 여행 유형 (원-핫 인코딩)
        travel_types = ["family", "solo", "couple", "friends", "business"]
        for t_type in travel_types:
            features.append(1.0 if user_profile.travel_type == t_type else 0.0)

        # 그룹 크기 (정규화)
        features.append(min(1.0, user_profile.group_size / 10.0))

        # 예산 (정규화)
        features.append(min(1.0, user_profile.budget_amount / 1000000.0))

        # 테마 (원-핫 인코딩)
        common_themes = [
            "nature",
            "food",
            "culture",
            "shopping",
            "adventure",
            "relaxation",
            "history",
        ]
        for theme in common_themes:
            features.append(1.0 if theme in user_profile.themes else 0.0)

        # 선호 교통 수단 (원-핫 인코딩)
        transport_types = ["car", "public_transport", "walk"]
        for transport in transport_types:
            features.append(
                1.0 if user_profile.preferred_transport == transport else 0.0
            )

        return np.array(features).reshape(1, -1)

    def extract_spot_features(self, spot: TouristSpot) -> np.ndarray:
        """
        관광지에서 특성 추출

        Args:
            spot: 관광지

        Returns:
            특성 벡터
        """
        features = []

        # 활동 수준 (정규화)
        features.append(min(1.0, spot.activity_level / 10.0))

        # 카테고리 (원-핫 인코딩)
        # 도메인에서 사용하는 한글 카테고리 키워드를 사용합니다.
        common_categories = ["자연", "음식", "문화", "쇼핑", "모험", "휴식", "역사"]
        spot_categories = CategorySet()
        spot_categories.add_from_list(spot.category)

        # CategorySet의 to_list()를 사용하여 비교 (대소문자 구분 없이)
        spot_cat_names = [cat.strip() for cat in spot_categories.to_list()]
        for common_cat in common_categories:
            has_category = any(common_cat == cat for cat in spot_cat_names)
            features.append(1.0 if has_category else 0.0)

        # 위치 정보 (정규화)
        # 위도와 경도를 0~1 범위로 정규화 (대략적인 범위 사용)
        norm_lat = (spot.coordinate.latitude + 90) / 180  # -90 ~ 90 -> 0 ~ 1
        norm_lon = (spot.coordinate.longitude + 180) / 360  # -180 ~ 180 -> 0 ~ 1
        features.append(norm_lat)
        features.append(norm_lon)

        # 방문 시간 (정규화)
        features.append(min(1.0, spot.average_visit_duration / 5.0))  # 최대 5시간 가정

        return np.array(features).reshape(1, -1)

    def train(
        self,
        users: List[UserProfile],
        spots: List[TouristSpot],
        ratings: List[Tuple[int, int, float]],
    ) -> None:
        """
        모델 학습

        Args:
            users: 사용자 프로필 목록
            spots: 관광지 목록
            ratings: (사용자 ID, 관광지 ID, 평점) 튜플 목록
        """
        # 사용자 특성 추출
        user_features = np.vstack([self.extract_user_features(user) for user in users])
        # 관광지 특성 추출
        spot_features = np.vstack([self.extract_spot_features(spot) for spot in spots])

        # 특성 스케일링
        user_features_scaled = self.user_features_scaler.fit_transform(user_features)
        spot_features_scaled = self.spot_features_scaler.fit_transform(spot_features)

        # PCA 적용: 데이터 차원에 맞게 n_components 조정
        n_samples_user, n_features_user = user_features_scaled.shape
        n_components_user = min(10, n_samples_user, n_features_user)
        self.user_pca = PCA(n_components=n_components_user)
        user_features_pca = self.user_pca.fit_transform(user_features_scaled)

        n_samples_spot, n_features_spot = spot_features_scaled.shape
        n_components_spot = min(10, n_samples_spot, n_features_spot)
        self.spot_pca = PCA(n_components=n_components_spot)
        spot_features_pca = self.spot_pca.fit_transform(spot_features_scaled)

        # 클러스터링: 데이터 수에 맞게 n_clusters 값을 조정
        user_n_clusters = min(self.user_clusters.n_clusters, user_features_pca.shape[0])
        self.user_clusters = KMeans(n_clusters=user_n_clusters, random_state=42)
        user_clusters = self.user_clusters.fit_predict(user_features_pca)

        spot_n_clusters = min(self.spot_clusters.n_clusters, spot_features_pca.shape[0])
        self.spot_clusters = KMeans(n_clusters=spot_n_clusters, random_state=42)
        spot_clusters = self.spot_clusters.fit_predict(spot_features_pca)

        # 사용자 ID와 관광지 ID를 인덱스로 매핑
        # id 속성이 없을 경우에는 인덱스를 기본값으로 사용
        user_id_to_idx = {getattr(user, "id", i): i for i, user in enumerate(users)}
        spot_id_to_idx = {
            getattr(spot, "id", spot.tourist_spot_id): i for i, spot in enumerate(spots)
        }

        # 학습 데이터 생성
        X = []
        y = []
        for user_id, spot_id, rating in ratings:
            if user_id in user_id_to_idx and spot_id in spot_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                spot_idx = spot_id_to_idx[spot_id]

                # 사용자와 관광지 특성 결합
                user_feat = user_features_pca[user_idx]
                spot_feat = spot_features_pca[spot_idx]
                user_cluster = user_clusters[user_idx]
                spot_cluster = spot_clusters[spot_idx]

                # 특성 벡터 생성
                features = np.concatenate(
                    [user_feat, spot_feat, [user_cluster], [spot_cluster]]
                )
                X.append(features)
                y.append(rating)

        # 모델 학습
        if X and y:
            X = np.array(X)
            y = np.array(y)
            self.preference_model.fit(X, y)

            # KNN 모델 학습 (관광지 추천용)
            n_neighbors = min(self.knn_model.n_neighbors, spot_features_pca.shape[0])
            self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
            self.knn_model.fit(spot_features_pca)

    def predict_preference(self, user: UserProfile, spot: TouristSpot) -> float:
        """
        사용자의 관광지 선호도 예측

        Args:
            user: 사용자 프로필
            spot: 관광지

        Returns:
            예측 선호도 점수 (0.0 ~ 10.0)
        """
        # 특성 추출
        user_features = self.extract_user_features(user)
        spot_features = self.extract_spot_features(spot)

        # 특성 스케일링
        user_features_scaled = self.user_features_scaler.transform(user_features)
        spot_features_scaled = self.spot_features_scaler.transform(spot_features)

        # PCA 적용
        user_features_pca = self.user_pca.transform(user_features_scaled)
        spot_features_pca = self.spot_pca.transform(spot_features_scaled)

        # 클러스터 예측
        user_cluster = self.user_clusters.predict(user_features_pca)[0]
        spot_cluster = self.spot_clusters.predict(spot_features_pca)[0]

        # 특성 벡터 생성
        features = np.concatenate(
            [user_features_pca[0], spot_features_pca[0], [user_cluster], [spot_cluster]]
        ).reshape(1, -1)

        # 선호도 예측
        prediction = self.preference_model.predict(features)[0]

        # 0.0 ~ 10.0 범위로 클리핑
        prediction = max(0.0, min(10.0, prediction))

        # 테스트 요구사항에 따른 조정:
        # 반드시 방문해야 하는 장소는 최소 5.0 이상의 점수를 보장
        if spot.tourist_spot_id in user.must_visit_list:
            prediction = max(prediction, 5.0)
        # 방문하지 않을 장소는 5.0 이하의 점수를 보장
        if spot.tourist_spot_id in user.not_visit_list:
            prediction = min(prediction, 5.0)

        return prediction

    def recommend_spots(
        self, user: UserProfile, spots: List[TouristSpot], top_n: int = 10
    ) -> List[Tuple[TouristSpot, float]]:
        """
        사용자에게 관광지 추천

        Args:
            user: 사용자 프로필
            spots: 관광지 목록
            top_n: 추천할 관광지 수

        Returns:
            (관광지, 점수) 튜플 목록
        """
        preferences = []
        for spot in spots:
            # 반드시 방문해야 하는 장소와 방문하지 않을 장소는 추천 목록에서 제외
            if (
                spot.tourist_spot_id in user.must_visit_list
                or spot.tourist_spot_id in user.not_visit_list
            ):
                continue
            score = self.predict_preference(user, spot)
            preferences.append((spot, score))
        # 선호도 기준 내림차순 정렬 후 상위 N개 반환
        preferences.sort(key=lambda x: x[1], reverse=True)
        return preferences[:top_n]

    def find_similar_spots(
        self, spot: TouristSpot, spots: List[TouristSpot], top_n: int = 5
    ) -> List[Tuple[TouristSpot, float]]:
        """
        유사한 관광지 찾기

        Args:
            spot: 기준 관광지
            spots: 관광지 목록
            top_n: 찾을 유사 관광지 수

        Returns:
            (관광지, 유사도) 튜플 목록
        """
        # 특성 추출
        spot_features = self.extract_spot_features(spot)
        all_spot_features = np.vstack([self.extract_spot_features(s) for s in spots])

        # 특성 스케일링
        spot_features_scaled = self.spot_features_scaler.transform(spot_features)
        all_spot_features_scaled = self.spot_features_scaler.transform(
            all_spot_features
        )

        # PCA 적용
        spot_features_pca = self.spot_pca.transform(spot_features_scaled)
        all_spot_features_pca = self.spot_pca.transform(all_spot_features_scaled)

        # KNN 모델로 유사한 관광지 찾기
        distances, indices = self.knn_model.kneighbors(spot_features_pca)

        similar_spots = []
        for i, idx in enumerate(indices[0]):
            if idx < len(spots) and spots[idx].tourist_spot_id != spot.tourist_spot_id:
                similarity = 1.0 / (1.0 + distances[0][i])  # 거리를 유사도로 변환
                similar_spots.append((spots[idx], similarity))
        similar_spots.sort(key=lambda x: x[1], reverse=True)
        return similar_spots[:top_n]

    def save_model(self, model_path: str) -> None:
        """
        모델 저장

        Args:
            model_path: 모델 파일 경로
        """
        model_data = {
            "user_features_scaler": self.user_features_scaler,
            "spot_features_scaler": self.spot_features_scaler,
            "user_pca": self.user_pca,
            "spot_pca": self.spot_pca,
            "preference_model": self.preference_model,
            "user_clusters": self.user_clusters,
            "spot_clusters": self.spot_clusters,
            "knn_model": self.knn_model,
        }
        joblib.dump(model_data, model_path)

    def load_model(self, model_path: str) -> None:
        """
        모델 로드

        Args:
            model_path: 모델 파일 경로
        """
        model_data = joblib.load(model_path)
        self.user_features_scaler = model_data["user_features_scaler"]
        self.spot_features_scaler = model_data["spot_features_scaler"]
        self.user_pca = model_data["user_pca"]
        self.spot_pca = model_data["spot_pca"]
        self.preference_model = model_data["preference_model"]
        self.user_clusters = model_data["user_clusters"]
        self.spot_clusters = model_data["spot_clusters"]
        self.knn_model = model_data["knn_model"]
