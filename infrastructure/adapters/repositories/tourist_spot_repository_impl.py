import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from domain.entities.tourist_spot import TouristSpot
from domain.repositories.tourist_spot_repository import TouristSpotRepository
from domain.value_objects.coordinate import Coordinate

load_dotenv()
DB_USERNAME = os.getenv("DATABASE_USER_NAME")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
DB_ENDPOINT = os.getenv("DATABASE_ENDPOINT")
DB_NAME = os.getenv("DATABASE_NAME")
DB_PORT = os.getenv("DATABASE_PORT", "3306")
DATABASE_URL = (
    f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_ENDPOINT}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class TouristSpotRepositoryImpl(TouristSpotRepository):
    """
    MySQL RDS를 이용한 관광지 레포지토리 구현체
    (다이어그램 상 존재하는 컬럼만 반영)

    실제 테이블 tourist_spots 예시:
        CREATE TABLE tourist_spots (
            tourist_spot_id BIGINT PRIMARY KEY,
            name VARCHAR(255),
            latitude DOUBLE,
            longitude DOUBLE,
            activity_level VARCHAR(255),
            address TEXT,
            business_status VARCHAR(255),
            category JSON,
            home_page VARCHAR(255),
            naver_booking_url VARCHAR(255),
            tel VARCHAR(255),
            thumbnail_url TEXT,
            thumbnail_urls JSON,
            travel_city_id BIGINT
        );
    """

    def __init__(self, session: Optional[Session] = None):
        if session:
            self.session = session
        else:
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.session = SessionLocal()

    def find_all(self) -> List[TouristSpot]:
        """
        모든 관광지 조회

        Returns:
            모든 관광지 객체 목록
        """
        try:
            query = text("SELECT * FROM tourist_spots")
            rows = self.session.execute(query).mappings().all()
            return [self._map_row_to_entity(r) for r in rows]
        except Exception as e:
            # 실제 환경에서는 로깅 처리
            logger.error(f"Error in find_all: {e}", exc_info=True)
            return []

    def find_by_id(self, tourist_spot_id: int) -> Optional[TouristSpot]:
        query = text(
            "SELECT * FROM tourist_spots WHERE tourist_spot_id = :tourist_spot_id"
        )
        row = (
            self.session.execute(query, {"tourist_spot_id": tourist_spot_id})
            .mappings()
            .first()
        )
        if row is None:
            return None
        return self._map_row_to_entity(row)

    def find_by_city_id(self, travel_city_id: int) -> List[TouristSpot]:
        query = text(
            "SELECT * FROM tourist_spots WHERE travel_city_id = :travel_city_id"
        )
        rows = (
            self.session.execute(query, {"travel_city_id": travel_city_id})
            .mappings()
            .all()
        )
        return [self._map_row_to_entity(r) for r in rows]

    def find_by_ids(self, tourist_spot_ids: List[int]) -> List[TouristSpot]:
        query = text(
            "SELECT * FROM tourist_spots WHERE tourist_spot_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        rows = self.session.execute(query, {"ids": tourist_spot_ids}).mappings().all()
        return [self._map_row_to_entity(r) for r in rows]

    def find_nearby(
        self, latitude: float, longitude: float, radius_km: float
    ) -> List[TouristSpot]:
        R = 6371
        query = text(
            f"""
            SELECT * FROM (
              SELECT *,
                {R} * ACOS(
                    COS(RADIANS(:lat)) * COS(RADIANS(latitude)) *
                    COS(RADIANS(longitude) - RADIANS(:lon)) +
                    SIN(RADIANS(:lat)) * SIN(RADIANS(latitude))
                ) AS distance
              FROM tourist_spots
            ) AS subquery
            WHERE distance <= :radius
        """
        )
        rows = (
            self.session.execute(
                query, {"lat": latitude, "lon": longitude, "radius": radius_km}
            )
            .mappings()
            .all()
        )
        return [self._map_row_to_entity(r) for r in rows]

    def find_by_category(self, categories: List[str]) -> List[TouristSpot]:
        """JSON 컬럼(category)에 특정 문자열이 포함된 행 검색."""
        if not categories:
            return []
        filters = []
        params = {}
        for i, cat in enumerate(categories):
            param_name = f"cat_{i}"
            # JSON 내에 cat 문자열이 포함되어 있는지 LIKE로 확인
            filters.append(f"category LIKE :{param_name}")
            params[param_name] = f'%"{cat}"%'
        condition = " OR ".join(filters)
        query_str = f"SELECT * FROM tourist_spots WHERE {condition}"
        query = text(query_str)
        rows = self.session.execute(query, params).mappings().all()
        return [self._map_row_to_entity(r) for r in rows]

    def save(self, spot: TouristSpot) -> TouristSpot:
        """INSERT or UPDATE 후 commit."""
        existing = self.find_by_id(spot.tourist_spot_id)
        if existing:
            query = text(
                """
                UPDATE tourist_spots
                SET
                    name = :name,
                    latitude = :latitude,
                    longitude = :longitude,
                    activity_level = :activity_level,
                    address = :address,
                    business_status = :business_status,
                    category = :category,
                    home_page = :home_page,
                    naver_booking_url = :naver_booking_url,
                    tel = :tel,
                    thumbnail_url = :thumbnail_url,
                    thumbnail_urls = :thumbnail_urls,
                    travel_city_id = :travel_city_id
                WHERE tourist_spot_id = :tourist_spot_id
            """
            )
        else:
            query = text(
                """
                INSERT INTO tourist_spots (
                    tourist_spot_id,
                    name,
                    latitude,
                    longitude,
                    activity_level,
                    address,
                    business_status,
                    category,
                    home_page,
                    naver_booking_url,
                    tel,
                    thumbnail_url,
                    thumbnail_urls,
                    travel_city_id
                ) VALUES (
                    :tourist_spot_id,
                    :name,
                    :latitude,
                    :longitude,
                    :activity_level,
                    :address,
                    :business_status,
                    :category,
                    :home_page,
                    :naver_booking_url,
                    :tel,
                    :thumbnail_url,
                    :thumbnail_urls,
                    :travel_city_id
                )
            """
            )

        params = self._map_entity_to_params(spot)
        self.session.execute(query, params)
        self.session.commit()
        return spot

    def _map_row_to_entity(self, row) -> TouristSpot:
        """DB Row -> TouristSpot 엔티티 변환."""
        return TouristSpot(
            tourist_spot_id=row["tourist_spot_id"],
            name=row["name"],
            coordinate=Coordinate(row["latitude"], row["longitude"]),
            # activity_level이 varchar이므로 문자열 처리
            activity_level=row["activity_level"],
            address=row["address"],
            # business_status는 다이어그램 기준 컬럼 존재
            business_status=row["business_status"],
            category=json.loads(row["category"]) if row["category"] else [],
            home_page=row["home_page"],
            naver_booking_url=row["naver_booking_url"],
            tel=row["tel"],
            thumbnail_url=row["thumbnail_url"],
            thumbnail_urls=(
                json.loads(row["thumbnail_urls"]) if row["thumbnail_urls"] else []
            ),
            travel_city_id=row["travel_city_id"],
        )

    def _map_entity_to_params(self, spot: TouristSpot) -> dict:
        """TouristSpot 엔티티 -> SQL 파라미터 맵핑."""
        return {
            "tourist_spot_id": spot.tourist_spot_id,
            "name": spot.name,
            "latitude": spot.coordinate.latitude,
            "longitude": spot.coordinate.longitude,
            "activity_level": spot.activity_level,  # varchar
            "address": spot.address,
            "business_status": spot.business_status,
            "category": json.dumps(spot.category) if spot.category else None,
            "home_page": spot.home_page,
            "naver_booking_url": spot.naver_booking_url,
            "tel": spot.tel,
            "thumbnail_url": spot.thumbnail_url,
            "thumbnail_urls": (
                json.dumps(spot.thumbnail_urls) if spot.thumbnail_urls else None
            ),
            "travel_city_id": spot.travel_city_id,
        }
