import os
import tempfile
import unittest

from infrastructure.adapters.ml.enhanced_clustering_algorithm import \
    EnhancedClusteringAlgorithm


# Dummy 엔티티 클래스 정의 (실제 도메인 엔티티와 동일한 인터페이스 제공)
class DummyCoordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude


class DummyTouristSpot:
    def __init__(
        self, tourist_spot_id: int, latitude: float, longitude: float, category=None
    ):
        self.tourist_spot_id = tourist_spot_id
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.category = category or []
        # 추가적으로 cluster_scoring_model 등에서 spot.id 속성에 접근하는 경우 대응
        self.id = tourist_spot_id


class DummyUserProfile:
    def __init__(self, must_visit_list=None, not_visit_list=None, themes=None):
        self.must_visit_list = must_visit_list or []
        self.not_visit_list = not_visit_list or []
        self.themes = themes or []


class TestEnhancedClusteringAlgorithm(unittest.TestCase):
    def setUp(self):
        # 두 그룹의 관광지를 생성
        # 그룹 A: (37.50, 126.90) 근방, 그룹 B: (37.60, 127.00) 근방
        self.spots = [
            DummyTouristSpot(1, 37.5000, 126.9000),
            DummyTouristSpot(2, 37.5005, 126.9005),
            DummyTouristSpot(3, 37.5010, 126.9010),
            DummyTouristSpot(4, 37.6000, 127.0000),
            DummyTouristSpot(5, 37.6005, 127.0005),
            DummyTouristSpot(6, 37.6010, 127.0010),
        ]
        # 사용자 프로필: 반드시 방문해야 하는 관광지와 방문 제외 관광지 설정
        self.user_profile = DummyUserProfile(
            must_visit_list=[2, 5],  # spot id 2와 5는 반드시 방문 대상
            not_visit_list=[3],  # spot id 3은 제외 대상
        )
        # 각 관광지의 기본 선호도 점수: (예시)
        self.base_scores = {1: 6.0, 2: 7.0, 3: 5.0, 4: 8.0, 5: 9.0, 6: 6.5}

        # EnhancedClusteringAlgorithm 인스턴스 생성 (내부에 ClusterScoringModel 기본 인스턴스 사용)
        self.algorithm = EnhancedClusteringAlgorithm()

    def test_find_optimal_clusters(self):
        # 최소 2개에서 최대 5개까지 검증
        optimal_clusters = self.algorithm.find_optimal_clusters(
            self.spots, min_clusters=2, max_clusters=5
        )
        self.assertIsInstance(optimal_clusters, int)
        self.assertGreaterEqual(optimal_clusters, 1)
        self.assertLessEqual(optimal_clusters, len(self.spots))

    def test_cluster_with_kmeans(self):
        num_clusters = 2
        clusters = self.algorithm.cluster_with_kmeans(self.spots, num_clusters)
        # 반환된 클러스터 개수가 num_clusters여야 함
        self.assertEqual(len(clusters), num_clusters)
        # 전체 관광지 수의 합은 입력과 같아야 함
        total_spots = sum(len(cluster) for cluster in clusters.values())
        self.assertEqual(total_spots, len(self.spots))

    def test_cluster_with_dbscan(self):
        clusters = self.algorithm.cluster_with_dbscan(
            self.spots, eps_km=10.0, min_samples=2
        )
        # DBSCAN의 경우 노이즈(-1) 클러스터가 있을 수 있으니 모든 그룹의 합이 전체 관광지 수와 같아야 함
        total_spots = sum(len(cluster) for cluster in clusters.values())
        self.assertEqual(total_spots, len(self.spots))
        # 반환된 clusters는 dict 타입이어야 함
        self.assertIsInstance(clusters, dict)

    def test_select_optimal_clusters(self):
        # 먼저, 클러스터링 수행 (K-means 기준)
        # 방문 제외 대상(not_visit_list)을 미리 필터링
        filtered_spots = [
            spot
            for spot in self.spots
            if spot.tourist_spot_id not in self.user_profile.not_visit_list
        ]

        num_clusters = 2
        clusters = self.algorithm.cluster_with_kmeans(filtered_spots, num_clusters)

        # num_days: 여행 일수 2일 가정
        num_days = 2

        # select_optimal_clusters는 내부에서 클러스터 점수 산출 후 선택하므로 결과는 리스트 형태
        selected = self.algorithm.select_optimal_clusters(
            clusters, self.user_profile, num_days, self.base_scores
        )
        self.assertIsInstance(selected, list)
        # 각 일자별로 할당된 클러스터(관광지 리스트)가 존재해야 함
        self.assertEqual(len(selected), num_days)
        for cluster in selected:
            self.assertIsInstance(cluster, list)
            # 방문 제외 대상이 필터링된 상태여야 함.
            for spot in cluster:
                self.assertNotIn(spot.tourist_spot_id, self.user_profile.not_visit_list)

    def test_cluster_with_user_preferences(self):
        num_days = 2
        clusters = self.algorithm.cluster_with_user_preferences(
            self.spots, self.user_profile, num_days, method="kmeans"
        )
        # 반환된 결과는 dict로, 키가 0부터 num_days-1까지 매핑됨
        self.assertIsInstance(clusters, dict)
        self.assertEqual(len(clusters), num_days)
        # 각 클러스터 내의 관광지들은 not_visit_list에 포함되지 않아야 함
        for cluster in clusters.values():
            for spot in cluster:
                self.assertNotIn(spot.tourist_spot_id, self.user_profile.not_visit_list)
        # 그리고 반드시 방문 관광지 (must_visit_list) 중 하나가 최소한 한 클러스터에 포함되면 좋음 (조건이 적용되었는지 확인)
        found_must_visit = any(
            any(
                spot.tourist_spot_id in self.user_profile.must_visit_list
                for spot in cluster
            )
            for cluster in clusters.values()
        )
        self.assertTrue(found_must_visit)

    def test_cluster_spots_for_multi_day_trip(self):
        num_days = 3
        clusters = self.algorithm.cluster_spots_for_multi_day_trip(
            self.spots, self.user_profile, num_days, self.base_scores
        )
        # 결과는 dict이며, 키는 0 ~ (num_days-1)
        self.assertIsInstance(clusters, dict)
        self.assertEqual(len(clusters), num_days)
        for day, cluster in clusters.items():
            self.assertIsInstance(cluster, list)
            # 제외 대상이 포함되지 않았는지 확인
            for spot in cluster:
                self.assertNotIn(spot.tourist_spot_id, self.user_profile.not_visit_list)

    def test_optimize_cluster_sequence(self):
        # 테스트용 클러스터 사전 생성: 각 클러스터는 리스트로 구성
        clusters = {
            0: [self.spots[0], self.spots[1]],
            1: [self.spots[3], self.spots[4]],
            2: [self.spots[5]],
        }
        num_days = 2
        sequence = self.algorithm.optimize_cluster_sequence(
            clusters, self.user_profile, num_days
        )
        # 반환된 sequence는 리스트이며, 길이는 num_days
        self.assertIsInstance(sequence, list)
        self.assertEqual(len(sequence), num_days)
        # 반환된 각 클러스터 id는 clusters의 키에 포함되어야 함
        for cluster_id in sequence:
            self.assertIn(cluster_id, clusters)

    def test_visualize_clusters(self):
        # 클러스터 시각화 테스트: clusters 생성 후 임시 파일에 저장한 후 파일 존재 여부 확인
        clusters = {
            0: [self.spots[0], self.spots[1]],
            1: [self.spots[3], self.spots[4]],
        }
        # 임시 파일 생성 (확장자 .png)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.close()
        output_path = temp_file.name

        try:
            self.algorithm.visualize_clusters(clusters, output_path=output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
        finally:
            os.remove(output_path)
