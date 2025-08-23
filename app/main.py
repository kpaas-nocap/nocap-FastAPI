from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

from free_news_similarity_analyzer import free_analyze_and_summarize
from premium_news_similarity_analyzer import premium_analyze_and_summarize

app = FastAPI(title="News Similarity API", version="1.0.0")

# 요청 DTO
class NewsDto(BaseModel):
    plan: str = "free"
    category: Optional[str] = None
    mainNewsDto: dict
    newsDtos: List[dict]

@app.post("/analyze")
async def analyze_news(dto: NewsDto):
    try:
        data = dto.dict()
        if data.get("plan") == "premium":
            return premium_analyze_and_summarize(data)
        else:
            return free_analyze_and_summarize(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))