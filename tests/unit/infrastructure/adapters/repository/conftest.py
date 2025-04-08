# conftest.py
import os
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pytest

@pytest.fixture
def in_memory_engine():
    engine = create_engine("sqlite:///:memory:")
    return engine

@pytest.fixture
def session(in_memory_engine):
    SessionLocal = sessionmaker(bind=in_memory_engine)
    session = SessionLocal()

    with in_memory_engine.connect() as conn:
        # 엔티티의 모든 필드에 맞춰 테이블 스키마를 구성
        conn.execute(text("""
            CREATE TABLE tourist_spots (
                tourist_spot_id INTEGER PRIMARY KEY,
                name VARCHAR(255),
                latitude DOUBLE,
                longitude DOUBLE,
                activity_level FLOAT,
                address VARCHAR(255),
                business_status VARCHAR(255),
                business_hours VARCHAR(255),
                category TEXT,
                home_page VARCHAR(255),
                naver_booking_url VARCHAR(255),
                tel VARCHAR(255),
                thumbnail_url TEXT,
                thumbnail_urls TEXT,
                travel_city_id INTEGER,
                average_visit_duration FLOAT,
                opening_time VARCHAR(255),
                closing_time VARCHAR(255),
                opening_hours VARCHAR(255)
            )
        """))
        conn.commit()

        json_path = os.path.join(os.path.dirname(__file__), "data", "test_place_data.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                # 실제 JSON 데이터에 없는 필드들은 기본값 지정
                params = {
                    "tourist_spot_id": int(item.get("id", 0)),
                    "name": item.get("name", ""),
                    "latitude": float(item.get("y", 0)),
                    "longitude": float(item.get("x", 0)),
                    "activity_level": 1.0,
                    "address": item.get("address", ""),
                    "business_status": (
                        item.get("businessStatus", {})
                            .get("status", {})
                            .get("description", "")
                    ),
                    # business_hours는 test_place_data.json 구조에 따라 적절히 매핑
                    "business_hours": item.get("businessHours", ""),
                    "category": json.dumps(item.get("category", []), ensure_ascii=False),
                    "home_page": item.get("homePage", ""),
                    "naver_booking_url": "",  # 데이터에 없으므로 빈 문자열
                    "tel": item.get("tel", ""),
                    "thumbnail_url": item.get("thumUrl", ""),
                    "thumbnail_urls": json.dumps(item.get("thumUrls", []), ensure_ascii=False),
                    "travel_city_id": 0,
                    "average_visit_duration": 1.0,
                    "opening_time": "",
                    "closing_time": "",
                    "opening_hours": ""
                }

                conn.execute(
                    text("""
                        INSERT INTO tourist_spots (
                            tourist_spot_id,
                            name,
                            latitude,
                            longitude,
                            activity_level,
                            address,
                            business_status,
                            business_hours,
                            category,
                            home_page,
                            naver_booking_url,
                            tel,
                            thumbnail_url,
                            thumbnail_urls,
                            travel_city_id,
                            average_visit_duration,
                            opening_time,
                            closing_time,
                            opening_hours
                        ) VALUES (
                            :tourist_spot_id,
                            :name,
                            :latitude,
                            :longitude,
                            :activity_level,
                            :address,
                            :business_status,
                            :business_hours,
                            :category,
                            :home_page,
                            :naver_booking_url,
                            :tel,
                            :thumbnail_url,
                            :thumbnail_urls,
                            :travel_city_id,
                            :average_visit_duration,
                            :opening_time,
                            :closing_time,
                            :opening_hours
                        )
                    """),
                    params
                )
            conn.commit()

    yield session
    session.close()