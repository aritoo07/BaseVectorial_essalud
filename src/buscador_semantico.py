import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class BuscadorSemanticoEsSalud:
    """
    Buscador semántico especializado para documentos EsSalud
    """
    
    def __init__(self, modelo='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.modelo = SentenceTransformer(modelo)
        self.indice = None
        self.fragmentos_documentos = []
        self.metadatos = []
        self.estadisticas = {}
    
    def cargar_base_datos(self, ruta_base="base_vectorial"):
        """Carga la base de datos vectorial de EsSalud"""
        try:
            self.indice = faiss.read_index(f"{ruta_base}/indice_vectorial.faiss")
            
            with open(f"{ruta_base}/fragmentos.pkl", 'rb') as f:
                self.fragmentos_documentos = pickle.load(f)
            
            with open(f"{ruta_base}/metadatos.pkl", 'rb') as f:
                self.metadatos = pickle.load(f)
            
            with open(f"{ruta_base}/estadisticas.pkl", 'rb') as f:
                self.estadisticas = pickle.load(f)
            
            print(f"🔍 BUSCADOR CARGADO - {self.indice.ntotal} vectores listos")
            return True
            
        except Exception as e:
            print(f"❌ ERROR cargando base: {str(e)}")
            return False
    
    def buscar_semantica(self, consulta, top_k=5, umbral_similitud=0.0):
        """
        Realiza búsqueda semántica en documentos EsSalud
        """
        if self.indice is None:
            print("❌ ERROR: Primero carga la base de datos")
            return []
        
        # Convertir consulta a vector
        vector_consulta = self.modelo.encode([consulta]).astype('float32')
        
        # 🟢 PASO CLAVE: Normalizar la consulta antes de buscar (¡Lo que faltaba!)
        faiss.normalize_L2(vector_consulta)
        
        # Buscar similares (FAISS IndexFlatIP usa similitud coseno)
        similitudes, indices = self.indice.search(vector_consulta, top_k)
        
        # Formatear resultados
        resultados = []
        for i, (similitud, idx) in enumerate(zip(similitudes[0], indices[0])):
            if idx < len(self.fragmentos_documentos) and similitud >= umbral_similitud:
                metadata = self.metadatos[idx] if idx < len(self.metadatos) else {}
                
                resultados.append({
                    'rank': i + 1,
                    'contenido': self.fragmentos_documentos[idx],
                    'similitud': float(similitud),  # Ya es similitud coseno (0-1)
                    'archivo': metadata.get('archivo', 'Desconocido'),
                    'tipo_documento': metadata.get('tipo_documento', 'General'),
                    'fragmento_id': metadata.get('fragmento_id', 0)
                })
        
        return resultados
    
    def buscar_por_tipo_documento(self, consulta, tipo_documento, top_k=5):
        """
        Búsqueda filtrada por tipo de documento
        """
        resultados = self.buscar_semantica(consulta, top_k=50)  # Buscar más resultados
        
        # Filtrar por tipo de documento
        resultados_filtrados = [
            r for r in resultados 
            if r['tipo_documento'].lower() == tipo_documento.lower()
        ]
        
        return resultados_filtrados[:top_k]
    
    def mostrar_resultados(self, resultados, consulta):
        """Muestra resultados de forma organizada"""
        print(f"\n🔍 RESULTADOS PARA: '{consulta}'")
        print("=" * 80)
        
        if not resultados:
            print("❌ No se encontraron resultados relevantes")
            return
        
        for resultado in resultados:
            print(f"\n📄 #{resultado['rank']} - {resultado['tipo_documento']}")
            print(f"📁 Archivo: {resultado['archivo']}")
            print(f"⭐ Similitud: {resultado['similitud']:.3f}")
            print(f"📝 Contenido: {resultado['contenido'][:200]}...")
            print("-" * 60)

# DEMO ESPECÍFICA PARA ESSALUD
def demo_busquedas_essalud():
    buscador = BuscadorSemanticoEsSalud()
    
    if not buscador.cargar_base_datos():
        return
    
    # Consultas típicas de EsSalud
    consultas_essalud = [
        "procedimiento de atención al asegurado",
        "requisitos para prestaciones económicas", 
        "normas de calidad en atención médica",
        "funciones del personal de salud",
        "derechos y deberes del asegurado"
    ]
    
    for consulta in consultas_essalud:
        resultados = buscador.buscar_semantica(consulta, top_k=3)
        buscador.mostrar_resultados(resultados, consulta)

if __name__ == "__main__":
    demo_busquedas_essalud()