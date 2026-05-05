"""
RAG Engine — vector knowledge base for attraction data.

Implements:
  - Document / RAGResult Pydantic models (task 2.3)
  - RAGEngine class with full Cache Hit/Miss logic (tasks 4.1 – 4.6)

Architecture:
  query()
    └─ _check_cache()          → Cache Hit  → _retrieve()
                               → Cache Miss → _fetch_and_store() → _retrieve()
    └─ generate_ai_notes()
    └─ return RAGResult
"""

from __future__ import annotations

import asyncio
import logging
import os
import textwrap
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mcp.server import MCPServer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "attractions"
_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_CHROMA_DIR = "./chroma_db"
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 50
_TOP_K = 5
_SOURCE_TIMEOUT = 10.0  # seconds per MCP source


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


# ---------------------------------------------------------------------------
# Helper: text chunking
# ---------------------------------------------------------------------------


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for attraction knowledge.

    Lifecycle
    ---------
    engine = RAGEngine(mcp_server)
    await engine.initialize()   # loads embedding model + ChromaDB
    result = await engine.query("金閣寺", city="京都", country="日本")
    """

    def __init__(self, mcp_server: "MCPServer") -> None:
        self._mcp = mcp_server
        self._collection = None  # set in initialize()
        self._embed_model = None  # SentenceTransformer, set in initialize()
        self._openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Async initialisation (task 4.1)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Load the embedding model and connect to ChromaDB.

        Must be called once before any query() calls.
        Runs the blocking model-load in a thread pool so it doesn't block
        the event loop.
        """
        await asyncio.get_event_loop().run_in_executor(None, self._sync_initialize)

    def _sync_initialize(self) -> None:
        """Blocking initialisation — runs in a thread pool executor."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        # ChromaDB persistent client (requirement 6.8)
        chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)

        # Get or create the attractions collection
        self._collection = chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity (requirement 6.6)
        )

        # Multilingual embedding model (requirement 6.1)
        self._embed_model = SentenceTransformer(_EMBED_MODEL)
        logger.info("RAGEngine initialised: collection=%s, model=%s", _COLLECTION_NAME, _EMBED_MODEL)

    # ------------------------------------------------------------------
    # Internal: embedding helper
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of text strings (blocking)."""
        if self._embed_model is None:
            raise RuntimeError("RAGEngine not initialised — call await engine.initialize() first.")
        vectors = self._embed_model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        """Non-blocking wrapper around _embed."""
        return await asyncio.get_event_loop().run_in_executor(None, self._embed, texts)

    # ------------------------------------------------------------------
    # Task 4.2 — Cache check
    # ------------------------------------------------------------------

    async def _check_cache(self, name: str, city: str, country: str) -> bool:
        """
        Return True (Cache Hit) if the attraction already has documents in
        ChromaDB, False (Cache Miss) otherwise.

        Key format: "{name.lower()}_{city.lower()}_{country.lower()}"
        (requirement 6.1, 6.2)
        """
        if self._collection is None:
            raise RuntimeError("RAGEngine not initialised.")

        attraction_key = f"{name.lower()}_{city.lower()}_{country.lower()}"

        def _query() -> int:
            results = self._collection.get(
                where={"attraction_key": attraction_key},
                limit=1,
            )
            return len(results.get("ids", []))

        count = await asyncio.get_event_loop().run_in_executor(None, _query)
        return count >= 1

    # ------------------------------------------------------------------
    # Task 4.3 — Fetch from MCP sources and store in ChromaDB
    # ------------------------------------------------------------------

    async def _fetch_and_store(
        self, name: str, city: str, country: str
    ) -> List[Document]:
        """
        Fetch attraction data from Wikipedia, Wikivoyage, and Nominatim via
        the MCP Server, chunk + embed the text, and persist to ChromaDB.

        Each source has a _SOURCE_TIMEOUT second timeout; failures are logged
        and skipped (requirement 6.9).  If ALL sources fail the method returns
        an empty list and the caller should fall back to LLM-only generation
        (requirement 6.7).
        """
        if self._collection is None:
            raise RuntimeError("RAGEngine not initialised.")

        attraction_key = f"{name.lower()}_{city.lower()}_{country.lower()}"
        documents: List[Document] = []
        nominatim_result = None  # kept for lat/lng extraction

        # ---- Wikipedia ------------------------------------------------
        try:
            wiki_result = await asyncio.wait_for(
                self._mcp.wikipedia_search(attraction_name=name),
                timeout=_SOURCE_TIMEOUT,
            )
            text = f"{wiki_result.title}\n\n{wiki_result.summary}\n\n{wiki_result.full_text}"
            for i, chunk in enumerate(_chunk_text(text)):
                documents.append(
                    Document(
                        id=f"{attraction_key}_wikipedia_{i}",
                        attraction_name=name,
                        city=city,
                        country=country,
                        source="wikipedia",
                        content=chunk,
                    )
                )
        except asyncio.TimeoutError:
            logger.warning("wikipedia_search timed out for %s — skipping", name)
        except Exception as exc:
            logger.warning("wikipedia_search failed for %s: %s — skipping", name, exc)

        # ---- Wikivoyage -----------------------------------------------
        try:
            wv_result = await asyncio.wait_for(
                self._mcp.wikivoyage_search(attraction_name=name, city=city),
                timeout=_SOURCE_TIMEOUT,
            )
            text = f"{wv_result.title}\n\n{wv_result.tips}"
            if wv_result.best_time:
                text += f"\n\nBest time to visit: {wv_result.best_time}"
            for i, chunk in enumerate(_chunk_text(text)):
                documents.append(
                    Document(
                        id=f"{attraction_key}_wikivoyage_{i}",
                        attraction_name=name,
                        city=city,
                        country=country,
                        source="wikivoyage",
                        content=chunk,
                    )
                )
        except asyncio.TimeoutError:
            logger.warning("wikivoyage_search timed out for %s — skipping", name)
        except Exception as exc:
            logger.warning("wikivoyage_search failed for %s: %s — skipping", name, exc)

        # ---- Nominatim ------------------------------------------------
        try:
            query = f"{name} {city} {country}".strip()
            nom_result = await asyncio.wait_for(
                self._mcp.nominatim_geocode(query=query, country=country),
                timeout=_SOURCE_TIMEOUT,
            )
            nominatim_result = nom_result
            text = (
                f"Location: {nom_result.display_name}\n"
                f"Coordinates: lat={nom_result.lat}, lng={nom_result.lng}\n"
                f"Type: {nom_result.osm_type} / {nom_result.category}"
            )
            for i, chunk in enumerate(_chunk_text(text)):
                documents.append(
                    Document(
                        id=f"{attraction_key}_nominatim_{i}",
                        attraction_name=name,
                        city=city,
                        country=country,
                        source="nominatim",
                        content=chunk,
                    )
                )
        except asyncio.TimeoutError:
            logger.warning("nominatim_geocode timed out for %s — skipping", name)
        except Exception as exc:
            logger.warning("nominatim_geocode failed for %s: %s — skipping", name, exc)

        # ---- Persist to ChromaDB (requirement 6.4) --------------------
        if documents:
            contents = [doc.content for doc in documents]
            embeddings = await self._embed_async(contents)

            for doc, emb in zip(documents, embeddings):
                doc.embedding = emb

            def _upsert() -> None:
                self._collection.upsert(
                    ids=[doc.id for doc in documents],
                    embeddings=[doc.embedding for doc in documents],
                    documents=[doc.content for doc in documents],
                    metadatas=[
                        {
                            "attraction_name": doc.attraction_name,
                            "city": doc.city,
                            "country": doc.country,
                            "source": doc.source,
                            "attraction_key": attraction_key,
                            "created_at": doc.created_at.isoformat(),
                        }
                        for doc in documents
                    ],
                )

            await asyncio.get_event_loop().run_in_executor(None, _upsert)
            logger.info(
                "Stored %d chunks for %s (key=%s)", len(documents), name, attraction_key
            )
        else:
            logger.warning(
                "All MCP sources failed for %s — will fall back to LLM-only generation", name
            )

        # Attach nominatim coordinates to the result set for the caller
        # (stored as a side-channel attribute so query() can pick them up)
        self._last_nominatim = nominatim_result
        return documents

    # ------------------------------------------------------------------
    # Task 4.4 — Semantic retrieval
    # ------------------------------------------------------------------

    async def _retrieve(self, query: str, top_k: int = _TOP_K) -> List[Document]:
        """
        Embed *query* and return the top-k most similar documents from
        ChromaDB using cosine similarity (requirement 6.2, 6.6).
        """
        if self._collection is None:
            raise RuntimeError("RAGEngine not initialised.")

        query_embedding = await self._embed_async([query])

        def _query_chroma() -> dict:
            return self._collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "embeddings"],
            )

        results = await asyncio.get_event_loop().run_in_executor(None, _query_chroma)

        documents: List[Document] = []
        ids = results.get("ids", [[]])[0]
        contents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0] or [[] for _ in ids]

        for doc_id, content, meta, emb in zip(ids, contents, metadatas, embeddings):
            documents.append(
                Document(
                    id=doc_id,
                    attraction_name=meta.get("attraction_name", ""),
                    city=meta.get("city", ""),
                    country=meta.get("country", ""),
                    source=meta.get("source", ""),
                    content=content,
                    embedding=emb if emb else [],
                )
            )

        return documents

    # ------------------------------------------------------------------
    # Task 4.5 — AI notes generation
    # ------------------------------------------------------------------

    async def generate_ai_notes(
        self,
        attraction_name: str,
        city: str,
        country: str,
        documents: List[Document],
        rag_failed: bool = False,
    ) -> str:
        """
        Generate AI travel notes for an attraction.

        If *documents* are available they are used as RAG context.
        If *rag_failed* is True (all sources failed) the LLM generates notes
        from its own knowledge and the result is labelled accordingly
        (requirement 6.5, 6.7).
        """
        if rag_failed or not documents:
            # Fallback: LLM-only generation (requirement 6.7)
            system_prompt = (
                "You are an expert travel guide. "
                "Provide concise, practical travel tips for the given attraction. "
                "Write in Traditional Chinese (繁體中文)."
            )
            user_prompt = (
                f"請為「{attraction_name}」（位於 {city}，{country}）"
                "提供實用的旅遊建議，包含最佳造訪時間、注意事項與在地小知識。"
                "請以繁體中文回答，約 150 字。"
            )
            suffix = "\n\n（資料來源：AI 生成）"
        else:
            # RAG-augmented generation (requirement 6.5)
            context_parts = [doc.content for doc in documents[:_TOP_K]]
            context = "\n\n---\n\n".join(context_parts)
            # Truncate context to avoid token overflow
            context = textwrap.shorten(context, width=3000, placeholder="...")

            system_prompt = (
                "You are an expert travel guide. "
                "Use the provided reference material to write concise, accurate travel tips. "
                "Write in Traditional Chinese (繁體中文)."
            )
            user_prompt = (
                f"參考資料：\n{context}\n\n"
                f"請根據以上資料，為「{attraction_name}」（位於 {city}，{country}）"
                "撰寫實用的旅遊備註，包含最佳造訪時間、注意事項與在地小知識。"
                "請以繁體中文回答，約 150 字。"
            )
            suffix = ""

        try:
            response = await self._openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            notes = response.choices[0].message.content.strip()
            return notes + suffix
        except Exception as exc:
            logger.error("generate_ai_notes failed for %s: %s", attraction_name, exc)
            return f"（AI 備註生成失敗：{exc}）（資料來源：AI 生成）"

    # ------------------------------------------------------------------
    # Task 4.6 — Public query method
    # ------------------------------------------------------------------

    async def query(
        self,
        attraction_name: str,
        city: str = "",
        country: str = "",
    ) -> RAGResult:
        """
        Main entry point for the Attraction_Agent.

        Flow:
          1. _check_cache()
             ├─ Cache Hit  → _retrieve()
             └─ Cache Miss → _fetch_and_store() → _retrieve()
          2. generate_ai_notes()
          3. Return RAGResult

        (requirement 6.1, 6.2, 6.3, 6.4, 6.5, 6.6)
        """
        self._last_nominatim = None  # reset side-channel

        # Step 1: Cache check
        cache_hit = await self._check_cache(attraction_name, city, country)

        rag_failed = False
        if cache_hit:
            # Cache Hit — retrieve directly (requirement 6.2)
            logger.info("Cache HIT for %s (%s, %s)", attraction_name, city, country)
            query_str = f"{attraction_name} {city} {country}".strip()
            documents = await self._retrieve(query_str)
        else:
            # Cache Miss — fetch, store, then retrieve (requirement 6.3, 6.4)
            logger.info("Cache MISS for %s (%s, %s) — fetching from MCP", attraction_name, city, country)
            documents = await self._fetch_and_store(attraction_name, city, country)

            if not documents:
                # All sources failed — mark degraded (requirement 6.7)
                rag_failed = True
            else:
                # Retrieve the freshly stored documents
                query_str = f"{attraction_name} {city} {country}".strip()
                documents = await self._retrieve(query_str)

        # Step 2: Generate AI notes
        ai_notes = await self.generate_ai_notes(
            attraction_name=attraction_name,
            city=city,
            country=country,
            documents=documents,
            rag_failed=rag_failed,
        )

        # Step 3: Extract coordinates from Nominatim side-channel (if available)
        lat: Optional[float] = None
        lng: Optional[float] = None
        category = "景點"
        if self._last_nominatim is not None:
            lat = self._last_nominatim.lat
            lng = self._last_nominatim.lng
            if self._last_nominatim.category:
                category = self._last_nominatim.category

        return RAGResult(
            attraction_name=attraction_name,
            cache_hit=cache_hit,
            documents=documents,
            ai_notes=ai_notes,
            image_urls=[],       # populated by Attraction_Agent via MCP
            rating=0.0,          # populated by Attraction_Agent
            review_count=0,      # populated by Attraction_Agent
            category=category,
            website_url=None,    # populated by Attraction_Agent
            lat=lat,
            lng=lng,
        )
