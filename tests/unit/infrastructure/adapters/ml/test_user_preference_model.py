import os
import tempfile
import unittest
from typing import List, Tuple

from infrastructure.adapters.ml.user_preference_model import \
    UserPreferenceModel


class DummyCoordinate:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude


class DummyTouristSpot:
    def __init__(
        self,
        tourist_spot_id: int,
        latitude: float,
        longitude: float,
        activity_level: float,
        category: List[str],
        average_visit_duration: float = 1.0,
    ):
        self.tourist_spot_id = tourist_spot_id
        self.coordinate = DummyCoordinate(latitude, longitude)
        self.activity_level = activity_level
        self.category = category
        self.average_visit_duration = average_visit_duration


class DummyUserProfile:
    def __init__(
        self,
        travel_type: str = "family",
        group_size: int = 2,
        budget_amount: float = 500000.0,
        themes: List[str] = None,
        preferred_transport: str = "car",
        must_visit_list: List[int] = None,
        not_visit_list: List[int] = None,
    ):
        self.travel_type = travel_type
        self.group_size = group_size
        self.budget_amount = budget_amount
        self.themes = themes or []
        self.preferred_transport = preferred_transport
        self.must_visit_list = must_visit_list or []
        self.not_visit_list = not_visit_list or []

    def generate_feature_vector(self):
        # 실제 UserProfile에서 하는 전처리가 있다면 넣으세요.
        pass


##################################
# Test Cases
##################################
class TestUserPreferenceModel(unittest.TestCase):
    def setUp(self):
        # 모델 인스턴스 준비
        self.model = UserPreferenceModel()

        # 가상 사용자 목록
        self.users = [
            DummyUserProfile(
                travel_type="family",
                group_size=3,
                budget_amount=300000,
                themes=["nature"],
                preferred_transport="car",
            ),
            DummyUserProfile(
                travel_type="solo",
                group_size=1,
                budget_amount=1000000,
                themes=["food", "culture"],
                preferred_transport="walk",
                must_visit_list=[200],
            ),
        ]

        # 가상 관광지 목록
        self.spots = [
            DummyTouristSpot(
                tourist_spot_id=100,
                latitude=37.0,
                longitude=127.0,
                activity_level=5.0,
                category=["자연", "역사"],
                average_visit_duration=2.0,
            ),
            DummyTouristSpot(
                tourist_spot_id=200,
                latitude=37.5,
                longitude=127.5,
                activity_level=7.0,
                category=["음식"],
                average_visit_duration=1.5,
            ),
            DummyTouristSpot(
                tourist_spot_id=300,
                latitude=38.0,
                longitude=128.0,
                activity_level=3.0,
                category=["쇼핑", "휴식"],
                average_visit_duration=2.5,
            ),
            DummyTouristSpot(
                tourist_spot_id=400,
                latitude=37.2,
                longitude=127.2,
                activity_level=8.0,
                category=["모험", "자연"],
                average_visit_duration=4.0,
            ),
        ]

        # 간단한 평점 데이터 (user_id, spot_id, rating)
        # user_id는 DummyUserProfile에 'id'가 없으므로 인덱스 0/1을 사용
        self.ratings: List[Tuple[int, int, float]] = [
            (0, 100, 7.0),
            (0, 200, 8.5),
            (1, 200, 9.0),
            (1, 300, 6.0),
            # user_idx=0, spot_id=400 에 대한 평점 추가
            (0, 400, 5.5),
        ]

    def test_train_and_predict(self):
        # 모델 학습
        self.model.train(self.users, self.spots, self.ratings)

        # 학습 후, preference_model과 knn_model이 fit되었는지 간단 검사
        self.assertIsNotNone(self.model.preference_model)
        self.assertIsNotNone(self.model.knn_model)

        # 임의 사용자/관광지 선호도 예측
        pred = self.model.predict_preference(self.users[0], self.spots[1])
        self.assertGreaterEqual(pred, 0.0)
        self.assertLessEqual(pred, 10.0)

    def test_must_visit_and_not_visit_adjustment(self):
        # 학습
        self.model.train(self.users, self.spots, self.ratings)

        # user_idx=1은 must_visit_list에 [200], not_visit_list없음
        # 만약 user_idx=1이 spot_id=200을 예측했을 때 결과가 반드시 5.0 이상인지 확인
        pred_must = self.model.predict_preference(
            self.users[1], self.spots[1]
        )  # spots[1].id=200
        self.assertGreaterEqual(pred_must, 5.0)

        # user_idx=0 에게 spot_id=300을 not_visit_list로 가정하고 재검증
        self.users[0].not_visit_list = [300]
        pred_not = self.model.predict_preference(
            self.users[0], self.spots[2]
        )  # spots[2].id=300
        self.assertLessEqual(pred_not, 5.0)

    def test_recommend_spots(self):
        self.model.train(self.users, self.spots, self.ratings)
        # user_idx=0에게 spots 추천
        recommendations = self.model.recommend_spots(self.users[0], self.spots, top_n=2)

        # 반환된 추천 목록은 최대 2개이며,
        self.assertLessEqual(len(recommendations), 2)
        # must_visit_list나 not_visit_list에 속한 관광지는 제외되어야 함
        # user_idx=0은 현재 not_visit_list=[300] (이전 테스트에서 설정), must_visit_list는 없음
        # 따라서 spot_id=300이 추천 목록에 없어야 한다
        rec_ids = [spot.tourist_spot_id for (spot, score) in recommendations]
        self.assertNotIn(300, rec_ids)

    def test_find_similar_spots(self):
        self.model.train(self.users, self.spots, self.ratings)
        # spot_id=100과 유사한 관광지를 찾음
        target_spot = self.spots[0]  # id=100
        similar = self.model.find_similar_spots(target_spot, self.spots, top_n=2)
        self.assertLessEqual(len(similar), 2)
        # 결과 내 spot 중 자기 자신(spot_id=100)은 없어야 함
        for s, sim_score in similar:
            self.assertNotEqual(s.tourist_spot_id, 100)
            self.assertGreaterEqual(sim_score, 0.0)

    def test_save_and_load_model(self):
        # 먼저 모델 학습
        self.model.train(self.users, self.spots, self.ratings)

        # 임시 파일
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_path = tmp.name

        try:
            # 모델 저장
            self.model.save_model(model_path)
            # 새 인스턴스를 만들어 로드
            new_model = UserPreferenceModel()
            new_model.load_model(model_path)

            # 로드된 모델로 예측 수행
            pred_old = self.model.predict_preference(self.users[0], self.spots[1])
            pred_new = new_model.predict_preference(self.users[0], self.spots[1])
            # 로드 전후 예측값이 거의 동일해야 함
            self.assertAlmostEqual(pred_old, pred_new, places=4)
        finally:
            os.remove(model_path)


if __name__ == "__main__":
    unittest.main()
