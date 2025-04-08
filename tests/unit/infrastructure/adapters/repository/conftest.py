"""
유닛테스트용 인메모리 sqlite 데이터베이스 fixture
"""

import json
from os import path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def in_memory_engine():
    """인메모리 sqlite 메모리 생성."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(in_memory_engine):
    """DB scheme 생성 및 json으로부터 더미데이터 주입"""
    session_local = sessionmaker(bind=in_memory_engine)
    session = session_local()

    with in_memory_engine.connect() as conn:
        conn.execute(
            text(
                """
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
                """
            )
        )
        conn.commit()

        json_path = path.join(
            path.dirname(__file__), "data", "test_place_data.json"
        )
        if path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            for item in data:
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
                    "business_hours": item.get("businessHours", ""),
                    "category": json.dumps(item.get("category", []), ensure_ascii=False),
                    "home_page": item.get("homePage", ""),
                    "naver_booking_url": "",
                    "tel": item.get("tel", ""),
                    "thumbnail_url": item.get("thumUrl", ""),
                    "thumbnail_urls": json.dumps(item.get("thumUrls", []), ensure_ascii=False),
                    "travel_city_id": 0,
                    "average_visit_duration": 1.0,
                    "opening_time": "",
                    "closing_time": "",
                    "opening_hours": "",
                }

                conn.execute(
                    text(
                        """
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
                        """
                    ),
                    params,
                )
            conn.commit()

    yield session
    session.close()
