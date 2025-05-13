import os
import tempfile
import unittest

import numpy as np

from infrastructure.adapters.ml.cluster_scoring_model import \
    ClusterScoringModel


# Dummy 클래스들을 정의 (실제 도메인 엔티티와 동일한 인터페이스가 필요함)
class DummyCoordinate:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
        # 테스트용 간단 유클리드 거리 계산
        return np.sqrt(
            (self.latitude - other.latitude) ** 2
            + (self.longitude - other.longitude) ** 2
        )


class DummyTouristSpot:
    def __init__(self, tourist_spot_id, activity_level, coordinate, category):
        self.tourist_spot_id = (
            tourist_spot_id  # 모델에서는 must_visit, not_visit 등에서 사용
        )
        self.activity_level = activity_level
        self.coordinate = coordinate
        self.category = category  # 리스트나 문자열 모두 가능
        self.id = tourist_spot_id  # rank_clusters에서 spot.id로도 접근


class DummyUserProfile:
    def __init__(self, themes, must_visit_list, not_visit_list):
        self.themes = themes  # 예: ["museum", "park"]
        self.must_visit_list = must_visit_list  # 반드시 방문해야 하는 관광지 id 리스트
        self.not_visit_list = not_visit_list  # 방문하지 않을 관광지 id 리스트


class TestClusterScoringModel(unittest.TestCase):
    def setUp(self):
        # Dummy UserProfile
        self.user_profile = DummyUserProfile(
            themes=["museum", "park"], must_visit_list=[1, 2], not_visit_list=[3]
        )

        # Dummy TouristSpots
        self.spots = [
            DummyTouristSpot(
                tourist_spot_id=1,
                activity_level=0.8,
                coordinate=DummyCoordinate(37.5665, 126.9780),
                category=["museum", "historical"],
            ),
            DummyTouristSpot(
                tourist_spot_id=2,
                activity_level=0.5,
                coordinate=DummyCoordinate(37.5651, 126.9895),
                category=["park"],
            ),
            DummyTouristSpot(
                tourist_spot_id=3,
                activity_level=0.6,
                coordinate=DummyCoordinate(37.5700, 126.9768),
                category=["shopping"],
            ),
        ]
        # 클러스터 중심 좌표: 여기서는 간단하게 첫번째 스팟의 좌표 사용
        self.center_coord = DummyCoordinate(37.5665, 126.9780)

        # 클러스터 데이터 (train 메서드에 전달할 데이터 구조)
        self.cluster_data = [
            {
                "spots": self.spots,
                "user_profile": self.user_profile,
                "center_coord": self.center_coord,
                "day_index": 0,
                "num_days": 3,
            }
        ]
        # 임의 평점 데이터
        self.ratings = [8.5]
        # 테스트 대상 모델 인스턴스 생성
        self.model = ClusterScoringModel()

    def test_extract_cluster_features(self):
        # 클러스터 특성 추출 테스트
        features = self.model.extract_cluster_features(
            spots=self.spots,
            user_profile=self.user_profile,
            center_coord=self.center_coord,
            day_index=0,
            num_days=3,
        )
        # extract_cluster_features에서는 총 8개의 피처를 생성함
        self.assertEqual(features.shape, (1, 8))

    def test_training_and_scoring(self):
        # 모델 학습 후 점수 예측 테스트
        self.model.train(self.cluster_data, self.ratings)
        score = self.model.score_cluster(
            spots=self.spots,
            user_profile=self.user_profile,
            center_coord=self.center_coord,
            day_index=0,
            num_days=3,
        )
        # 예측 점수는 0.0 ~ 10.0 범위여야 함
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_rank_clusters(self):
        # 클러스터 순위 산출 테스트
        # 여러 클러스터를 구성 (cluster id: 0, 1)
        clusters = {0: self.spots, 1: self.spots[:2]}  # 두 개의 관광지
        # 각 스팟에 대한 기본 선호도 점수 (tourist_spot_id 기반)
        base_scores = {1: 7.0, 2: 8.0, 3: 6.0}
        # features_scaler가 아직 학습되지 않았다면, rank_clusters에서는 base_score 평균을 사용하게 됨
        ranked_clusters = self.model.rank_clusters(
            clusters, self.user_profile, num_days=3, base_scores=base_scores
        )
        # 결과는 점수 내림차순으로 정렬된 (클러스터 id, 점수) 튜플 리스트여야 함
        self.assertTrue(
            all(
                ranked_clusters[i][1] >= ranked_clusters[i + 1][1]
                for i in range(len(ranked_clusters) - 1)
            )
        )

    def test_optimize_cluster_sequence(self):
        # 클러스터 최적 순서 계산 테스트
        clusters = {0: self.spots, 1: self.spots[:2], 2: self.spots[1:]}
        # 모델 학습(혹은 features_scaler의 fit 여부가 판단 기준)
        self.model.train(self.cluster_data, self.ratings)
        sequence = self.model.optimize_cluster_sequence(
            clusters, self.user_profile, num_days=2
        )
        # 최적 순서는 num_days(=2) 개의 클러스터 id를 반환해야 함
        self.assertEqual(len(sequence), 2)
        # 반환된 클러스터 id는 입력 clusters의 키 중 하나여야 함
        self.assertTrue(all(cluster_id in clusters for cluster_id in sequence))

    def test_save_and_load_model(self):
        # 모델 저장 및 불러오기 테스트
        self.model.train(self.cluster_data, self.ratings)
        # 임시 파일을 이용하여 모델 저장
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name
        try:
            self.model.save_model(model_path)
            # 새 인스턴스를 생성하고 저장한 모델을 불러옴
            new_model = ClusterScoringModel(model_path=model_path)
            # 로드된 features_scaler에는 이미 fit된 상태(예: mean_ 속성 존재)가 되어야 함
            self.assertTrue(hasattr(new_model.features_scaler, "mean_"))
        finally:
            os.remove(model_path)


if __name__ == "__main__":
    unittest.main()
