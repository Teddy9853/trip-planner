"""
MCP Server — data models and server skeleton.

Defines the Pydantic models for all MCP tool results (task 2.4).
Full tool implementations follow in task 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import quote

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic wrapper returned by every MCP tool call
# ---------------------------------------------------------------------------

class MCPToolResult(BaseModel):
    """Standardised envelope for every MCP tool response (requirement 5.6)."""

    tool_name: str = Field(..., description="Name of the tool called")
    success: bool = Field(..., description="Whether the tool call succeeded")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Return data on success")
    error_code: Optional[str] = Field(default=None, description="Error code on failure")
    error_message: Optional[str] = Field(default=None, description="Error description on failure")


# ---------------------------------------------------------------------------
# Tool-specific result models
# ---------------------------------------------------------------------------

class WikipediaResult(BaseModel):
    """Parsed result from the wikipedia_search tool."""

    title: str = Field(..., description="Wikipedia page title")
    summary: str = Field(..., description="Summary paragraph")
    full_text: str = Field(..., description="Full page text (HTML stripped)")
    url: str = Field(..., description="Wikipedia page URL")
    categories: List[str] = Field(default_factory=list, description="Page category tags")


class WikivoyageResult(BaseModel):
    """Parsed result from the wikivoyage_search tool."""

    title: str = Field(..., description="Wikivoyage page title")
    tips: str = Field(..., description="Travel tips and local recommendations")
    best_time: str = Field(default="", description="Best time to visit")
    url: str = Field(..., description="Wikivoyage page URL")


class NominatimResult(BaseModel):
    """Parsed result from the nominatim_geocode tool."""

    display_name: str = Field(..., description="Full address display name")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    lng: float = Field(..., ge=-180.0, le=180.0, description="Longitude")
    osm_type: str = Field(..., description="OSM object type (node / way / relation)")
    category: str = Field(default="", description="Attraction type (from OSM tag)")
    address: Dict[str, Any] = Field(default_factory=dict, description="Structured address fields")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

import httpx


class MCPServer:
    """
    Model Context Protocol server that dispatches tool calls to the
    appropriate implementation methods (requirements 5.4, 5.5, 5.6).
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient = httpx.AsyncClient()

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MCPServer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Dispatcher (requirement 5.4 / 5.6)
    # ------------------------------------------------------------------

    async def call_tool(self, tool_name: str, params: dict) -> MCPToolResult:
        """
        Route a tool call to the correct implementation method.

        Returns a structured MCPToolResult.  Unknown tool names are handled
        gracefully by returning an error result (requirement 5.5).
        Any exception raised by a tool is caught and returned as a structured
        error (requirement 5.6).
        """
        if tool_name == "wikipedia_search":
            try:
                result = await self.wikipedia_search(**params)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    data=result.model_dump(),
                )
            except Exception as e:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error_code="TOOL_ERROR",
                    error_message=str(e),
                )
        elif tool_name == "wikivoyage_search":
            try:
                result = await self.wikivoyage_search(**params)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    data=result.model_dump(),
                )
            except Exception as e:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error_code="TOOL_ERROR",
                    error_message=str(e),
                )
        elif tool_name == "nominatim_geocode":
            try:
                result = await self.nominatim_geocode(**params)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    data=result.model_dump(),
                )
            except Exception as e:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error_code="TOOL_ERROR",
                    error_message=str(e),
                )
        elif tool_name == "get_attraction_images":
            try:
                image_urls = await self.get_attraction_images(**params)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    data={"image_urls": image_urls},
                )
            except Exception as e:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error_code="TOOL_ERROR",
                    error_message=str(e),
                )
        else:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error_code="UNKNOWN_TOOL",
                error_message=f"Unknown tool: {tool_name}",
            )

    # ------------------------------------------------------------------
    # Tool stubs (fully implemented in subtasks 3.2 – 3.5)
    # ------------------------------------------------------------------

    async def wikipedia_search(
        self, attraction_name: str, language: str = "en"
    ) -> WikipediaResult:
        """Fetch and parse a Wikipedia page for the given attraction.

        Uses the MediaWiki opensearch API first to resolve the correct English
        page title (handles non-ASCII / non-English input names), then fetches
        the REST summary and HTML for that resolved title.
        """
        timeout = httpx.Timeout(10.0)
        base_url = f"https://{language}.wikipedia.org"

        # Step 1: resolve the best-matching English page title via opensearch
        search_resp = await self._client.get(
            f"{base_url}/w/api.php",
            params={
                "action": "opensearch",
                "search": attraction_name,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            timeout=timeout,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        titles = search_data[1] if len(search_data) > 1 else []
        if not titles:
            raise ValueError(f"Wikipedia: no results for '{attraction_name}'")

        resolved_title = titles[0]
        encoded_title = quote(resolved_title.replace(" ", "_"), safe="")
        rest_base = f"{base_url}/api/rest_v1/page"

        # Step 2: fetch summary for the resolved title
        summary_resp = await self._client.get(
            f"{rest_base}/summary/{encoded_title}", timeout=timeout
        )
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()

        title = summary_data.get("title", resolved_title)
        summary = summary_data.get("extract", "")
        categories_raw = summary_data.get("categories", [])
        if categories_raw and isinstance(categories_raw[0], dict):
            categories = [c.get("title", "") for c in categories_raw]
        else:
            categories = [str(c) for c in categories_raw]

        # Step 3: fetch full HTML and extract clean text
        html_resp = await self._client.get(
            f"{rest_base}/html/{encoded_title}", timeout=timeout
        )
        html_resp.raise_for_status()
        soup = BeautifulSoup(html_resp.text, "html.parser")
        full_text = soup.get_text(separator=" ", strip=True)

        page_url = f"{base_url}/wiki/{encoded_title}"

        return WikipediaResult(
            title=title,
            summary=summary,
            full_text=full_text,
            url=page_url,
            categories=categories,
        )

    async def wikivoyage_search(
        self, attraction_name: str, city: str
    ) -> WikivoyageResult:
        """Fetch and parse a Wikivoyage page for the given attraction.

        Uses opensearch to resolve the correct English page title first
        (handles non-ASCII / non-English input names), then falls back to
        city-level lookup if no direct match is found.
        """
        timeout = httpx.Timeout(10.0)
        base_url = "https://en.wikivoyage.org"

        def _extract_best_time(text: str) -> str:
            """Return the first sentence that mentions a climate/season keyword."""
            keywords = ["best time", "climate", "when to visit", "season"]
            sentences = text.split(".")
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            return sentence.strip() + "."
            return ""

        async def _opensearch(query: str) -> str | None:
            """Return the first opensearch result title, or None."""
            resp = await self._client.get(
                f"{base_url}/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            titles = data[1] if len(data) > 1 else []
            return titles[0] if titles else None

        async def _fetch_page(page_title: str) -> WikivoyageResult | None:
            """Fetch REST summary for a resolved page title."""
            encoded = quote(page_title.replace(" ", "_"), safe="")
            url = f"{base_url}/api/rest_v1/page/summary/{encoded}"
            resp = await self._client.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            tips = data.get("extract", "")
            title = data.get("title", page_title)
            best_time = _extract_best_time(tips)
            page_url = f"{base_url}/wiki/{encoded}"
            return WikivoyageResult(title=title, tips=tips, best_time=best_time, url=page_url)

        # Attempt 1: opensearch with attraction_name
        resolved = await _opensearch(attraction_name)
        if resolved:
            result = await _fetch_page(resolved)
            if result is not None:
                return result

        # Attempt 2: opensearch with city
        resolved = await _opensearch(city)
        if resolved:
            result = await _fetch_page(resolved)
            if result is not None:
                return result

        raise ValueError(
            f"Wikivoyage page not found for: {attraction_name}, {city}"
        )

    async def nominatim_geocode(
        self, query: str, country: str = ""
    ) -> NominatimResult:
        """Query OSM Nominatim for coordinates and address data.

        Tries the query as-is first.  If no results are returned (common when
        the name is in a non-Latin script), retries with just the city/country
        portion stripped from the query to let Nominatim's own search handle
        transliteration.
        """
        timeout = httpx.Timeout(5.0)
        headers = {"User-Agent": "travel-planner-enhanced/1.0 (educational project)"}

        async def _search(q: str) -> list:
            params: dict = {
                "q": q,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
            }
            if country:
                params["countrycodes"] = country
            resp = await self._client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()

        results = await _search(query)

        # If the full query (which may contain CJK characters) returns nothing,
        # split off the first token (attraction name) and retry with just the
        # remainder (city + country), which is more likely to be in Latin script.
        if not results:
            parts = query.split(None, 1)
            if len(parts) > 1:
                results = await _search(parts[1])

        if not results:
            raise ValueError(f"Nominatim: no results for query: {query}")

        r = results[0]
        return NominatimResult(
            display_name=r.get("display_name", ""),
            lat=float(r["lat"]),
            lng=float(r["lon"]),
            osm_type=r.get("osm_type", ""),
            category=r.get("type", ""),
            address=r.get("address", {}),
        )

    async def get_attraction_images(
        self, attraction_name: str, max_images: int = 5
    ) -> list[str]:
        """Return a list of image URLs from Wikimedia Commons.

        Uses a two-step approach:
        1. Search for files related to the attraction in the File namespace.
        2. Fetch imageinfo for each result to get the direct URL.

        Returns at most max_images URLs filtered to common image formats.
        Returns an empty list on any error or when no images are found.
        """
        timeout = httpx.Timeout(10.0)
        base_url = "https://commons.wikimedia.org/w/api.php"

        # Step 1: search for files
        search_params = {
            "action": "query",
            "list": "search",
            "srnamespace": 6,  # File namespace
            "srsearch": attraction_name,
            "srlimit": max_images * 2,  # fetch extra to account for filtering
            "format": "json",
        }
        try:
            search_resp = await self._client.get(base_url, params=search_params, timeout=timeout)
            search_resp.raise_for_status()
            search_data = search_resp.json()
        except Exception:
            return []

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []

        # Step 2: get image URLs for each result
        image_urls = []
        allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}

        for item in search_results:
            if len(image_urls) >= max_images:
                break
            title = item.get("title", "")
            if not title:
                continue
            try:
                info_params = {
                    "action": "query",
                    "titles": title,
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "format": "json",
                }
                info_resp = await self._client.get(base_url, params=info_params, timeout=timeout)
                info_resp.raise_for_status()
                info_data = info_resp.json()
                pages = info_data.get("query", {}).get("pages", {})
                for page in pages.values():
                    imageinfo = page.get("imageinfo", [])
                    if imageinfo:
                        url = imageinfo[0].get("url", "")
                        if url:
                            ext = "." + url.rsplit(".", 1)[-1].lower() if "." in url else ""
                            if ext in allowed_exts:
                                image_urls.append(url)
            except Exception:
                continue  # skip this image on error, try next

        return image_urls[:max_images]
