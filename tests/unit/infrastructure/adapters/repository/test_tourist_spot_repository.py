import pytest

from domain.entities.tourist_spot import TouristSpot
from domain.value_objects.coordinate import Coordinate
from infrastructure.adapters.repositories.tourist_spot_repository_impl import \
    TouristSpotRepositoryImpl


@pytest.fixture
def repo(session):
    return TouristSpotRepositoryImpl(session=session)


def test_find_by_id_in_memory(repo, session):
    """테스트 데이터에서 ID가 11583210인 레코드를 조회한다."""
    spot = repo.find_by_id(11583210)
    assert spot is not None
    assert spot.name == "제주신라호텔"
    assert abs(spot.coordinate.latitude - 33.2473930) < 0.0001


def test_update_in_memory(repo, session):
    """테스트 데이터에서 ID가 11583210인 레코드를 수정하여 저장한 후, 변경사항이 반영되었는지 확인한다."""
    spot = repo.find_by_id(11583210)
    assert spot is not None
    spot.name = "제주신라호텔 업데이트"
    repo.save(spot)
    updated = repo.find_by_id(11583210)
    assert updated.name == "제주신라호텔 업데이트"


def test_insert_in_memory(repo, session):
    """새로운 레코드를 삽입하고, 데이터가 정상 삽입되었는지 확인한다."""
    new_spot = TouristSpot(
        tourist_spot_id=99999999,
        name="신규 호텔 테스트",
        coordinate=Coordinate(35.0, 128.0),
        activity_level=3,
        address="테스트 주소",
        business_hours="10:00-21:00",
        category=["테스트"],
        home_page="https://test.com",
        naver_booking_url="https://book.test.com",
        tel="010-0000-0000",
        thumbnail_url="https://test.com/img.jpg",
        thumbnail_urls=["https://test.com/img1.jpg"],
        travel_city_id=0,
        average_visit_duration=60,
        opening_hours="매일",
        opening_time="09:00:00",
        closing_time="18:00:00",
    )
    repo.save(new_spot)
    inserted = repo.find_by_id(99999999)
    assert inserted is not None
    assert inserted.name == "신규 호텔 테스트"


def test_find_by_city_id_in_memory(repo, session):
    """travel_city_id가 0인 레코드를 조회한다."""
    spots = repo.find_by_city_id(0)
    assert len(spots) > 0
    for spot in spots:
        assert spot.travel_city_id == 0


def test_find_by_ids_in_memory(repo, session):
    """ID 목록을 이용해 여러 레코드를 조회한다."""
    ids = [11583210, 38278713]
    spots = repo.find_by_ids(ids)
    returned_ids = {spot.tourist_spot_id for spot in spots}
    assert set(ids).issubset(returned_ids)


def test_find_nearby_in_memory(repo, session):
    """주어진 좌표와 반경 내에 있는 레코드를 조회한다."""
    new_spot = TouristSpot(
        tourist_spot_id=77777777,
        name="근접 호텔 테스트",
        coordinate=Coordinate(33.247, 126.978),
        activity_level=1,
        address="근접 주소",
        business_hours="09:00-18:00",
        category=["근접"],
        home_page="",
        naver_booking_url="",
        tel="",
        thumbnail_url="",
        thumbnail_urls=[],
        travel_city_id=0,
        average_visit_duration=0,
        opening_hours="",
        opening_time="",
        closing_time="",
    )
    repo.save(new_spot)
    nearby_spots = repo.find_nearby(33.247, 126.978, radius_km=1)
    ids = [spot.tourist_spot_id for spot in nearby_spots]
    assert 77777777 in ids


def test_find_by_category_in_memory(repo, session):
    """특정 카테고리를 포함하는 레코드를 조회한다."""
    spots = repo.find_by_category(["숙박", "호텔"])
    found = any("숙박" in spot.category for spot in spots)
    assert found


def test_find_by_city_id_no_result(repo, session):
    """존재하지 않는 travel_city_id로 조회시 빈 리스트를 반환하는지 확인한다."""
    spots = repo.find_by_city_id(999999)
    assert spots == []


def test_find_by_ids_no_result(repo, session):
    """존재하지 않는 ID 목록으로 조회시 빈 리스트를 반환하는지 확인한다."""
    spots = repo.find_by_ids([999999, 888888])
    assert spots == []


def test_find_nearby_no_result(repo, session):
    """주어진 좌표 및 반경에서 일치하는 레코드가 없을 경우 빈 리스트를 반환하는지 확인한다."""
    spots = repo.find_nearby(0, 0, radius_km=1)
    assert spots == []


def test_find_by_category_no_result(repo, session):
    """존재하지 않는 카테고리로 조회시 빈 리스트를 반환하는지 확인한다."""
    spots = repo.find_by_category(["존재하지 않는 카테고리"])
    assert spots == []
