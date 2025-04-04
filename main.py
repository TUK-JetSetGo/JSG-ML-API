"""
메인 애플리케이션 모듈
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    uvicorn.run(app, host="0.0.0.0")
