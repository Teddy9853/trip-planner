"""
RAG Engine — data models and engine skeleton.

Defines the Pydantic models used by the RAG pipeline (task 2.3).
Full engine implementation follows in task 4.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Internal document model (stored in / retrieved from ChromaDB)
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A single text chunk stored in the vector knowledge base."""

    id: str = Field(..., description="Unique document identifier (attraction_key + chunk index)")
    attraction_name: str = Field(..., description="景點名稱")
    city: str = Field(default="", description="所在城市")
    country: str = Field(default="", description="所在國家")
    source: str = Field(
        ...,
        description="資料來源：wikipedia | wikivoyage | nominatim",
    )
    content: str = Field(..., description="文字內容（已分段）")
    embedding: List[float] = Field(default_factory=list, description="向量表示")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間（UTC）")


# ---------------------------------------------------------------------------
# RAG query result returned to Attraction_Agent
# ---------------------------------------------------------------------------

class RAGResult(BaseModel):
    """Aggregated result returned by RAGEngine.query()."""

    attraction_name: str = Field(..., description="景點名稱")
    cache_hit: bool = Field(..., description="是否命中快取")
    documents: List[Document] = Field(default_factory=list, description="檢索到的相關文件")
    ai_notes: str = Field(default="", description="由 LLM 生成的 AI 備註文字")
    image_urls: List[str] = Field(default_factory=list, max_length=5, description="景點圖片 URL（最多 5 張）")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="景點評分")
    review_count: int = Field(default=0, ge=0, description="評論數量")
    category: str = Field(default="景點", description="景點類型")
    website_url: Optional[str] = Field(default=None, description="官方網站 URL")
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="緯度")
    lng: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="經度")
