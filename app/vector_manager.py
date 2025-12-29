import faiss
import numpy as np
import pickle
import time
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import os

class VectorDatabaseManager:
    """
    Gestor para base de datos vectorial de EsSalud
    """
    
    def __init__(self, db_path: str, model_name: str):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        self._load_database()
    
    def _load_database(self):
        try:
            index_path = os.path.join(self.db_path, "indice_vectorial.faiss")
            self.index = faiss.read_index(index_path)
            
            docs_path = os.path.join(self.db_path, "fragmentos.pkl")
            with open(docs_path, 'rb') as f: self.documents = pickle.load(f)
            
            meta_path = os.path.join(self.db_path, "metadatos.pkl")
            with open(meta_path, 'rb') as f: self.metadata = pickle.load(f)
            
            print(f"✅ Base vectorial cargada: {self.index.ntotal} vectores.")
            
        except Exception as e:
            print(f"❌ Error cargando base vectorial: {str(e)}")
            raise
    
    def semantic_search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3, filters: Dict = None) -> Tuple[List[Dict], float]:
        start_time = time.time()
        try:
            # 1. Generar embedding
            query_embedding = self.model.encode([query]).astype('float32')
            
            # 🟢 CORRECCIÓN: Normalizar la consulta también
            # Esto asegura que la similitud coseno (IndexFlatIP) sea siempre <= 1.0
            faiss.normalize_L2(query_embedding)
            
            # 2. Buscar
            similarities, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.documents) and similarity >= similarity_threshold:
                    if filters and not self._apply_filters(self.metadata[idx], filters):
                        continue
                    
                    result = {
                        'rank': i + 1,
                        'content': self.documents[idx],
                        'similarity': float(similarity), # Ahora será máx 1.0
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                        'source': self.metadata[idx].get('archivo', 'Desconocido') if idx < len(self.metadata) else 'Desconocido'
                    }
                    results.append(result)
            
            return results, time.time() - start_time
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {str(e)}")
            return [], time.time() - start_time
    
    def _apply_filters(self, metadata: Dict, filters: Dict) -> bool:
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True