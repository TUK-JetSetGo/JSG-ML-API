import json
import os
from datetime import time
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
print("DB_USERNAME:", os.getenv("DATABASE_USERNAME"))
DATABASE_URL = (
    f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_ENDPOINT}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class TouristSpotRepositoryImpl(TouristSpotRepository):
    """
    MySQL RDS를 이용한 관광지 레포지토리 구현체
    컬럼명과 엔티티(TouristSpot) 필드를 일치시켰다.
    DB 스키마 예시:
        CREATE TABLE tourist_spots (
            tourist_spot_id BIGINT PRIMARY KEY,
            name VARCHAR(255),
            latitude DOUBLE,
            longitude DOUBLE,
            activity_level FLOAT,
            address VARCHAR(255),
            business_hours VARCHAR(255),
            category JSON,
            home_page VARCHAR(255),
            naver_booking_url VARCHAR(255),
            tel VARCHAR(255),
            thumbnail_url TEXT,
            thumbnail_urls JSON,
            travel_city_id BIGINT,
            average_visit_duration FLOAT,
            opening_hours VARCHAR(255),
            opening_time TIME,
            closing_time TIME
        );
    """

    def __init__(self, session: Optional[Session] = None):
        if session:
            self.session = session
        else:
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.session = SessionLocal()

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
        if not categories:
            return []
        filters = []
        params = {}
        for i, cat in enumerate(categories):
            param_name = f"cat_{i}"
            filters.append(f"category LIKE :{param_name}")
            params[param_name] = f'%"{cat}"%'
        condition = " OR ".join(filters)
        query_str = f"SELECT * FROM tourist_spots WHERE {condition}"
        query = text(query_str)
        rows = self.session.execute(query, params).mappings().all()
        return [self._map_row_to_entity(r) for r in rows]

    def save(self, spot: TouristSpot) -> TouristSpot:
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
                    business_hours = :business_hours,
                    category = :category,
                    home_page = :home_page,
                    naver_booking_url = :naver_booking_url,
                    tel = :tel,
                    thumbnail_url = :thumbnail_url,
                    thumbnail_urls = :thumbnail_urls,
                    travel_city_id = :travel_city_id,
                    average_visit_duration = :average_visit_duration,
                    opening_hours = :opening_hours,
                    opening_time = :opening_time,
                    closing_time = :closing_time
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
                    business_hours,
                    category,
                    home_page,
                    naver_booking_url,
                    tel,
                    thumbnail_url,
                    thumbnail_urls,
                    travel_city_id,
                    average_visit_duration,
                    opening_hours,
                    opening_time,
                    closing_time
                ) VALUES (
                    :tourist_spot_id,
                    :name,
                    :latitude,
                    :longitude,
                    :activity_level,
                    :address,
                    :business_hours,
                    :category,
                    :home_page,
                    :naver_booking_url,
                    :tel,
                    :thumbnail_url,
                    :thumbnail_urls,
                    :travel_city_id,
                    :average_visit_duration,
                    :opening_hours,
                    :opening_time,
                    :closing_time
                )
            """
            )

        params = self._map_entity_to_params(spot)
        self.session.execute(query, params)
        self.session.commit()
        return spot

    def _map_row_to_entity(self, row) -> TouristSpot:
        return TouristSpot(
            tourist_spot_id=row["tourist_spot_id"],
            name=row["name"],
            coordinate=Coordinate(row["latitude"], row["longitude"]),
            activity_level=row["activity_level"],
            address=row["address"],
            business_hours=row["business_hours"],
            category=json.loads(row["category"]) if row["category"] else [],
            home_page=row["home_page"],
            naver_booking_url=row["naver_booking_url"],
            tel=row["tel"],
            thumbnail_url=row["thumbnail_url"],
            thumbnail_urls=(
                json.loads(row["thumbnail_urls"]) if row["thumbnail_urls"] else []
            ),
            travel_city_id=row["travel_city_id"],
            average_visit_duration=row["average_visit_duration"],
            opening_hours=row["opening_hours"],
            opening_time=(
                time.fromisoformat(row["opening_time"]) if row["opening_time"] else None
            ),
            closing_time=(
                time.fromisoformat(row["closing_time"]) if row["closing_time"] else None
            ),
        )

    def _map_entity_to_params(self, spot: TouristSpot) -> dict:
        return {
            "tourist_spot_id": spot.tourist_spot_id,
            "name": spot.name,
            "latitude": spot.coordinate.latitude,
            "longitude": spot.coordinate.longitude,
            "activity_level": spot.activity_level,
            "address": spot.address,
            "business_hours": spot.business_hours,
            "category": json.dumps(spot.category) if spot.category else None,
            "home_page": spot.home_page,
            "naver_booking_url": spot.naver_booking_url,
            "tel": spot.tel,
            "thumbnail_url": spot.thumbnail_url,
            "thumbnail_urls": (
                json.dumps(spot.thumbnail_urls) if spot.thumbnail_urls else None
            ),
            "travel_city_id": spot.travel_city_id,
            "average_visit_duration": spot.average_visit_duration,
            "opening_hours": spot.opening_hours,
            "opening_time": (
                spot.opening_time.isoformat() if spot.opening_time else None
            ),
            "closing_time": (
                spot.closing_time.isoformat() if spot.closing_time else None
            ),
        }
