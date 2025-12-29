import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Configuración de la API
    API_TITLE = "ESsalud RAG API"
    API_DESCRIPTION = "API de Búsqueda Semántica para Documentos de EsSalud"
    API_VERSION = "1.0.0"
    
    # Configuración de la Base Vectorial
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "base_vectorial")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Configuración LLM (Generación)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # NUEVA CONFIGURACIÓN PARA GEMINI
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "openai") # Proveedor por defecto
    
    # Configuración de Búsqueda
    DEFAULT_TOP_K = 5
    SIMILARITY_THRESHOLD = 0.3

settings = Settings()