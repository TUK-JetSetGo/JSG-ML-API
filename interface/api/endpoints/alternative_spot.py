"""
대체 여행지 추천 API 엔드포인트 모듈
"""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from application.usecase.alternative_spot_usecase import (
    AlternativeSpotRequest, AlternativeSpotResponse, AlternativeSpotUseCase)
from domain.services.alternative_spot_service import AlternativeSpotService
from infrastructure.adapters.repositories.tourist_spot_repository_impl import \
    TouristSpotRepositoryImpl


# 요청 모델
class AlternativeSpotRequestModel(BaseModel):
    """대체 여행지 추천 요청 모델"""

    itinerary: List[int] = Field(
        ..., description="현재 여행 일정에 포함된 관광지 ID 목록"
    )
    modify_idx: List[int] = Field(
        ..., description="대체 여행지를 추천받을 인덱스 목록 (0부터 시작)"
    )
    radius: float = Field(5.0, description="대체 여행지 검색 반경 (km)")
    recommend_count: int = Field(5, description="각 인덱스별 추천할 대체 여행지 개수")


# 응답 모델
class AlternativeSpotResponseModel(BaseModel):
    """대체 여행지 추천 응답 모델"""

    alternatives: Dict[str, List[int]] = Field(
        ..., description="인덱스별 대체 여행지 ID 목록을 담은 딕셔너리"
    )


# 의존성 주입
def get_alternative_spot_usecase() -> AlternativeSpotUseCase:
    """대체 여행지 추천 유스케이스 의존성 주입"""

    tourist_spot_repository = TouristSpotRepositoryImpl()
    alternative_spot_service = AlternativeSpotService()

    return AlternativeSpotUseCase(
        tourist_spot_repository=tourist_spot_repository,
        alternative_spot_service=alternative_spot_service,
    )


# 라우터 정의
router = APIRouter(prefix="/api/v1/alternatives", tags=["alternatives"])


@router.post(
    "/recommend",
    response_model=AlternativeSpotResponseModel,
    summary="대체 여행지 추천",
    description="입력으로 관광지 번호 리스트가 들어오면 그 여행지와 가까운 대체 여행지를 추천합니다.",
)
async def recommend_alternative_spots(
    request_data: AlternativeSpotRequestModel,
    usecase: AlternativeSpotUseCase = Depends(get_alternative_spot_usecase),
) -> AlternativeSpotResponseModel:
    """
    대체 여행지 추천 엔드포인트

    Args:
        request_data: 요청 데이터
        usecase: 대체 여행지 추천 유스케이스

    Returns:
        대체 여행지 추천 결과
    """
    try:
        # 유스케이스 요청 객체 생성
        usecase_request = AlternativeSpotRequest(
            itinerary=request_data.itinerary,
            modify_idx=request_data.modify_idx,
            radius=request_data.radius,
            recommend_count=request_data.recommend_count,
        )

        # 유스케이스 실행
        response: AlternativeSpotResponse = usecase.execute(usecase_request)

        # 응답 모델 변환
        return AlternativeSpotResponseModel(alternatives=response.alternatives)

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(ex)}")
