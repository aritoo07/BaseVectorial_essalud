from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.config import settings

# --- Modelos de Búsqueda de Documentos ---

class SearchResult(BaseModel):
    """Representa un fragmento de documento encontrado."""
    rank: int = Field(..., description="Posición en la lista de resultados.")
    content: str = Field(..., description="Contenido del fragmento del documento.")
    similarity: float = Field(..., description="Puntuación de similitud con la consulta (0.0 a 1.0).")
    metadata: Dict[str, Any] = Field(..., description="Metadatos del fragmento.")
    source: str = Field(..., description="Nombre del archivo fuente.")

class SearchRequest(BaseModel):
    query: str = Field(..., description="La pregunta o consulta a buscar.")
    top_k: int = Field(10, description="Número de documentos.", ge=1, le=50)

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int = 0
    search_time: float = Field(..., description="Tiempo total de búsqueda.")

# --- Modelos de Generación (RAG) ---

class Message(BaseModel):
    """Representa un mensaje en el historial del chat."""
    role: str = Field(..., description="Rol del emisor: 'user' o 'assistant'.")
    content: str = Field(..., description="Contenido del mensaje.")

class GenerateRequest(BaseModel):
    """Esquema de entrada para RAG con historial."""
    query: str = Field(..., description="La pregunta actual.")
    # 🟢 NUEVO: Historial de chat para contexto
    history: List[Message] = Field(default=[], description="Historial de conversación reciente.")
    
    top_k: int = Field(10, description="Documentos de contexto.", ge=1, le=50)
    max_tokens: int = Field(2048, description="Límite de tokens.", ge=128, le=8192)
    provider: Optional[str] = Field(None, description="Proveedor LLM.")

class GenerateResponse(BaseModel):
    query: str = Field(..., description="La consulta (posiblemente reescrita).")
    answer: str = Field(..., description="Respuesta generada.")
    sources: List[SearchResult] = Field(..., description="Fuentes utilizadas.")
    generation_time: float = Field(..., description="Tiempo total.")