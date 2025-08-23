from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

from free_news_similarity_analyzer import free_analyze_and_summarize
from premium_news_similarity_analyzer import premium_analyze_and_summarize

app = FastAPI(title="뉴스 기사 유사도 분석 및 요약 API", version="1.0.0")

# CORS 설정 (배포 시 외부 도메인 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 DTO
class NewsDto(BaseModel):
    plan: str = "free"
    category: Optional[str] = None
    mainNewsDto: dict
    newsDtos: List[dict]

@app.post("/analyze", description="뉴스 데이터를 분석하고 요약 결과 반환합니다.")
async def analyze_news(dto: NewsDto):
    """
    plan: free/premium
    category: 뉴스 카테고리
    mainNewsDto: 주 뉴스 데이터
    newsDtos: 비교 뉴스 리스트
    """
    try:
        data = dto.dict()
        if data.get("plan") == "premium":
            return premium_analyze_and_summarize(data)
        else:
            return free_analyze_and_summarize(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health", description="FastAPI 서버의 헬스체크 API입니다.")
async def health_check():
    return {"status" : "ok"}