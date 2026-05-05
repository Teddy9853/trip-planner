"""
MCP Server — data models and server skeleton.

Defines the Pydantic models for all MCP tool results (task 2.4).
Full tool implementations follow in task 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic wrapper returned by every MCP tool call
# ---------------------------------------------------------------------------

class MCPToolResult(BaseModel):
    """Standardised envelope for every MCP tool response (requirement 5.6)."""

    tool_name: str = Field(..., description="呼叫的工具名稱")
    success: bool = Field(..., description="工具呼叫是否成功")
    data: Optional[Dict[str, Any]] = Field(default=None, description="成功時的回傳資料")
    error_code: Optional[str] = Field(default=None, description="失敗時的錯誤代碼")
    error_message: Optional[str] = Field(default=None, description="失敗時的錯誤描述")


# ---------------------------------------------------------------------------
# Tool-specific result models
# ---------------------------------------------------------------------------

class WikipediaResult(BaseModel):
    """Parsed result from the wikipedia_search tool."""

    title: str = Field(..., description="Wikipedia 頁面標題")
    summary: str = Field(..., description="摘要段落")
    full_text: str = Field(..., description="完整頁面文字（已清理 HTML）")
    url: str = Field(..., description="Wikipedia 頁面 URL")
    categories: List[str] = Field(default_factory=list, description="頁面分類標籤")


class WikivoyageResult(BaseModel):
    """Parsed result from the wikivoyage_search tool."""

    title: str = Field(..., description="Wikivoyage 頁面標題")
    tips: str = Field(..., description="旅遊貼士與在地建議")
    best_time: str = Field(default="", description="最佳造訪時間")
    url: str = Field(..., description="Wikivoyage 頁面 URL")


class NominatimResult(BaseModel):
    """Parsed result from the nominatim_geocode tool."""

    display_name: str = Field(..., description="完整地址顯示名稱")
    lat: float = Field(..., ge=-90.0, le=90.0, description="緯度")
    lng: float = Field(..., ge=-180.0, le=180.0, description="經度")
    osm_type: str = Field(..., description="OSM 物件類型（node / way / relation）")
    category: str = Field(default="", description="景點類型（來自 OSM tag）")
    address: Dict[str, Any] = Field(default_factory=dict, description="結構化地址欄位")
