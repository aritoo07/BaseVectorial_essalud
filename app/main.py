from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time
import os
from typing import List

from app.config import settings
from app.models import SearchResult, GenerateRequest, GenerateResponse, SearchRequest, SearchResponse
from app.vector_manager import VectorDatabaseManager 
from app.llm_service import LLMService

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inicialización ---
print(f"🔄 Verificando base vectorial en: {settings.VECTOR_DB_PATH}...")
vector_manager = None
if os.path.exists(settings.VECTOR_DB_PATH):
    try:
        vector_manager = VectorDatabaseManager(settings.VECTOR_DB_PATH, settings.EMBEDDING_MODEL)
        print("✅ Base vectorial cargada.")
    except Exception as e:
        print(f"❌ Error base vectorial: {e}")

llm_service = None
try:
    llm_service = LLMService()
    print("✅ Servicio LLM inicializado.")
except Exception as e:
    print(f"❌ Error LLM: {e}")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "system": "EsSalud RAG AI"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    if not vector_manager or not llm_service:
        raise HTTPException(status_code=503, detail="Servicios no inicializados.")

    start_time = time.time()
    
    try:
        # 1. Convertir historial a dicts simples
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
        
        # 2. Contextualizar la pregunta (IA entiende "y cuáles son?" basándose en lo anterior)
        search_query = llm_service.contextualize_query(request.query, history_dicts)
        
        # 3. Búsqueda Vectorial (AUMENTADA A TOP_K=15 para más contexto)
        # Si la pregunta es muy corta (ej. "Hola"), bajamos el top_k para no gastar recursos
        actual_top_k = request.top_k if len(search_query) > 10 else 3
        
        search_results, _ = vector_manager.semantic_search(
            query=search_query, 
            top_k=actual_top_k
        )
        
        # 4. Generación de Respuesta con Historial
        answer = llm_service.generate_answer(
            query=request.query, # Pasamos la original al prompt para que fluya natural
            context_documents=search_results,
            history=history_dicts, # Pasamos el historial completo
            max_tokens=request.max_tokens,
            provider=request.provider 
        )
        
        # 5. Formatear fuentes
        sources = [
            SearchResult(
                rank=r['rank'],
                content=r['content'][:200] + "...", 
                similarity=r['similarity'],
                metadata=r['metadata'],
                source=r['source']
            ) for r in search_results
        ]
        
        return GenerateResponse(
            query=search_query,
            answer=answer,
            sources=sources,
            generation_time=time.time() - start_time
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)