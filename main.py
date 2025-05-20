"""
메인 애플리케이션 모듈
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interface.api.endpoints import itinerary, alternative_spot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="여행 동선 최적화 API",
    description="사용자 맞춤형 여행 일정 최적화 및 추천 API",
    version="1.0.0",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(itinerary.router)
app.include_router(alternative_spot.router)


@app.get("/")
async def root():
    """
    루트 엔드포인트

    Returns:
        API 정보
    """
    return {
        "name": "여행 동선 최적화 API",
        "version": "0.1.0",
        "description": "사용자 맞춤형 여행 일정 최적화 및 추천 API",
        "endpoints": [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",  # ← module_name:app_name 형태의 import string
        host="0.0.0.0",
        port=8000,
        reload=True,  # 코드 변경 시 자동 재시작 활성화
        log_level="debug",  # 터미널에 debug/info 로그 출력
    )
