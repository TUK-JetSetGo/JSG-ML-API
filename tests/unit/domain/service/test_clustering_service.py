"""
ClusteringService에 대한 유닛테스트
"""

import math

import numpy as np

from domain.services.clustering_service import ClusteringService


class DummyCoordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def distance_to(self, other: "DummyCoordinate") -> float:
        return math.sqrt(
            (self.latitude - other.latitude) ** 2
            + (self.longitude - other.longitude) ** 2
        )


class DummyTouristSpot:
    def __init__(
        self,
        spot_id: int,
        latitude: float,
        longitude: float,
        average_visit_duration: float = 1.0,
    ):
        self.id = spot_id  # pylint: disable=invalid-name
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.average_visit_duration = average_visit_duration


def test_cluster_tourist_spots():
    """
    관광지 목록에 대해 KMeans 클러스터링이 올바르게 수행되는지 테스트합니다.
    - 입력으로 4개의 DummyTouristSpot을 주고, num_clusters를 2로 지정합니다.
    - 반환된 클러스터 수가 2이고, 모든 관광지가 할당되어 있는지 확인합니다.
    """
    spots = [
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 0.1),
        DummyTouristSpot(3, 10.0, 10.0),
        DummyTouristSpot(4, 10.1, 10.0),
    ]
    clusters = ClusteringService.cluster_tourist_spots(spots, num_clusters=2)
    assert isinstance(clusters, dict)
    # 클러스터의 수는 입력된 num_clusters와 같아야 함.
    assert len(clusters) == 2
    # 모든 관광지가 클러스터에 포함되어 있어야 함.
    total_spots = sum(len(cluster) for cluster in clusters.values())
    assert total_spots == len(spots)


def test_calculate_cluster_center():
    """
    주어진 관광지 목록의 중심 좌표가 평균 좌표와 일치하는지 테스트합니다.
    """
    spots = [
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 2.0),
        DummyTouristSpot(3, 2.0, 0.0),
    ]
    center = ClusteringService.calculate_cluster_center(spots)
    expected_lat = (0.0 + 0.0 + 2.0) / 3
    expected_lon = (0.0 + 2.0 + 0.0) / 3
    assert math.isclose(center.latitude, expected_lat, rel_tol=1e-5)
    assert math.isclose(center.longitude, expected_lon, rel_tol=1e-5)


def test_calculate_cluster_centers():
    """
    클러스터 딕셔너리를 입력하면, 각 클러스터의 중심 좌표가 올바르게 계산되는지 확인합니다.
    """
    cluster1 = [DummyTouristSpot(1, 0.0, 0.0), DummyTouristSpot(2, 0.0, 2.0)]
    cluster2 = [DummyTouristSpot(3, 2.0, 0.0), DummyTouristSpot(4, 2.0, 2.0)]
    clusters = {0: cluster1, 1: cluster2}
    centers = ClusteringService.calculate_cluster_centers(clusters)
    # cluster1 center: (0.0, 1.0), cluster2 center: (2.0, 1.0)
    assert math.isclose(centers[0].latitude, 0.0, rel_tol=1e-5)
    assert math.isclose(centers[0].longitude, 1.0, rel_tol=1e-5)
    assert math.isclose(centers[1].latitude, 2.0, rel_tol=1e-5)
    assert math.isclose(centers[1].longitude, 1.0, rel_tol=1e-5)


def test_calculate_cluster_radius():
    """
    주어진 클러스터 내 관광지 중 중심과 가장 먼 거리(반경)이 올바르게 계산되는지 테스트합니다.
    """
    spots = [
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 3.0),
        DummyTouristSpot(3, 4.0, 0.0),
    ]
    center = DummyCoordinate(1.3333, 1.0)  # 임의의 중심 (테스트용)
    radius = ClusteringService.calculate_cluster_radius(spots, center)
    # 각 spot과 중심 간 거리를 직접 계산하여 최대값 검증
    distances = [
        center.distance_to(
            DummyCoordinate(s.coordinate.latitude, s.coordinate.longitude)
        )
        for s in spots
    ]
    expected_radius = max(distances)
    assert math.isclose(radius, expected_radius, rel_tol=1e-5)


def test_calculate_cluster_density():
    """
    클러스터 밀도가 (관광지 수 / 면적) 공식에 따라 올바르게 계산되는지 테스트합니다.
    """
    spots = [
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 1.0),
        DummyTouristSpot(3, 1.0, 0.0),
    ]
    radius_km = 10.0
    density = ClusteringService.calculate_cluster_density(spots, radius_km)
    expected_area = np.pi * (radius_km**2)
    expected_density = len(spots) / expected_area
    assert math.isclose(density, expected_density, rel_tol=1e-5)


def test_calculate_inter_cluster_distances():
    """
    두 클러스터 간 중심 좌표의 거리를 계산합니다.
    """
    cluster1 = [DummyTouristSpot(1, 0.0, 0.0), DummyTouristSpot(2, 0.0, 2.0)]
    cluster2 = [DummyTouristSpot(3, 2.0, 0.0), DummyTouristSpot(4, 2.0, 2.0)]
    clusters = {0: cluster1, 1: cluster2}
    distances = ClusteringService.calculate_inter_cluster_distances(clusters)

    # cluster1 center: (0, 1), cluster2 center: (2, 1)
    # Haversine 공식으로 두 중심 간의 예상 거리를 계산
    import math

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # 지구 반경 (km)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    expected_distance = haversine(0, 1, 2, 1)
    # 두 클러스터 간 거리가 서로 반대 방향으로도 저장되어야 함.
    assert (
        0,
        1,
    ) in distances, f"Expected key (0,1) in distances, got {distances.keys()}"
    assert (
        1,
        0,
    ) in distances, f"Expected key (1,0) in distances, got {distances.keys()}"
    assert math.isclose(
        distances[(0, 1)], expected_distance, rel_tol=1e-2
    ), f"Expected {expected_distance} km, got {distances[(0, 1)]} km"


def test_cluster_spots_kmeans():
    """
    cluster_spots_kmeans가 cluster_tourist_spots와 동일한 결과를 반환하는지 테스트합니다.
    """
    spots = [
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 0.1),
        DummyTouristSpot(3, 10.0, 10.0),
        DummyTouristSpot(4, 10.1, 10.0),
    ]
    clusters1 = ClusteringService.cluster_tourist_spots(spots, num_clusters=2)
    clusters2 = ClusteringService.cluster_spots_kmeans(spots, num_clusters=2)
    # 두 결과에서 전체 관광지 수가 동일한지 확인
    total1 = sum(len(lst) for lst in clusters1.values())
    total2 = sum(len(lst) for lst in clusters2.values())
    assert total1 == total2 == len(spots)


def test_cluster_spots_dbscan():
    """
    DBSCAN 기반 클러스터링이 eps 및 min_samples 조건에 따라 올바르게 그룹화되는지 테스트합니다.
    - 가까운 관광지들은 같은 클러스터에 속하고, outlier로 판단되는 관광지는 -1 번 클러스터로 분류됩니다.
    """
    # 3개의 관광지는 매우 가까이 있고, 1개의 outlier가 있음.
    spots = [
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 0.001),
        DummyTouristSpot(3, 0.001, 0.0),
        DummyTouristSpot(4, 0.1, 0.1),  # outlier
    ]
    clusters = ClusteringService.cluster_spots_dbscan(spots, eps_km=1.0, min_samples=2)
    # 주요 군집은 0번 클러스터, 노이즈는 -1 클러스터로 분류됨.
    # outlier가 -1 클러스터에 포함되어 있는지 확인합니다.
    assert -1 in clusters, f"DBSCAN 결과에 노이즈 클러스터 (-1)가 없어 {clusters}"
    # 주요 클러스터에 3개의 관광지가 있는지 확인
    main_cluster = [lst for key, lst in clusters.items() if key != -1]
    total_main = sum(len(lst) for lst in main_cluster)
    assert total_main == 3


def test_extract_coordinates():
    """
    관광지 목록으로부터 좌표 배열이 올바른 numpy array 형태로 추출되는지 테스트합니다.
    """
    spots = [DummyTouristSpot(1, 10.0, 20.0), DummyTouristSpot(2, 30.0, 40.0)]
    coords = ClusteringService.extract_coordinates(spots)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (2, 2)
    np.testing.assert_allclose(coords, np.array([[10.0, 20.0], [30.0, 40.0]]))


def test_find_optimal_clusters():
    """
    silhouette score를 기반으로 최적의 클러스터 수를 찾는지 테스트합니다.
    명확하게 두 개의 군집으로 나뉘는 데이터셋을 사용하여, optimal cluster 수가 2로 결정되는지 확인합니다.
    """
    spots = [
        # Cluster 1: around (0,0)
        DummyTouristSpot(1, 0.0, 0.0),
        DummyTouristSpot(2, 0.0, 0.1),
        DummyTouristSpot(3, 0.1, 0.0),
        # Cluster 2: around (10,10)
        DummyTouristSpot(4, 10.0, 10.0),
        DummyTouristSpot(5, 10.0, 10.1),
        DummyTouristSpot(6, 10.1, 10.0),
    ]
    optimal = ClusteringService.find_optimal_clusters(
        spots, min_clusters=2, max_clusters=3
    )
    # 데이터셋의 구조상 클러스터 수 2가 더 적절할 것으로 예상됩니다.
    assert optimal == 2, f"Expected optimal clusters 2, got {optimal}"
