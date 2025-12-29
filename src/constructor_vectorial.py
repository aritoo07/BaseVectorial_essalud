import os
import faiss
import pickle
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
import time

# Suprimimos advertencias técnicas
warnings.filterwarnings("ignore")

class BaseVectorialEsSalud:
    
    def __init__(self, modelo='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        print("🏥 INICIANDO CONSTRUCTOR VECTORIAL (MODO: LOCAL + OCR OPTIMIZADO)")
        self.modelo = SentenceTransformer(modelo)
        self.indice = None
        self.fragmentos_documentos = []
        self.metadatos = []
        
        try:
            self.ocr_engine = RapidOCR()
            print("👁️  Motor OCR activado. (Filtrará logos pequeños para velocidad)")
        except Exception as e:
            print(f"⚠️  No se pudo iniciar OCR: {e}")
            self.ocr_engine = None

    def _extraer_texto_con_ocr(self, ruta_pdf):
        texto_completo = ""
        doc = fitz.open(ruta_pdf)
        total_paginas = len(doc)
        
        print(f"   📄 Analizando {total_paginas} páginas...")
        
        for numero_pagina, pagina in enumerate(doc):
            # 1. Extraer texto digital normal (rápido)
            texto_pagina = pagina.get_text()
            
            # 2. Buscar imágenes GRANDES (Gráficos/Tablas) e ignorar LOGOS
            imagenes = pagina.get_images(full=True)
            texto_ocr = ""
            
            if imagenes and self.ocr_engine:
                for img_index, img in enumerate(imagenes):
                    xref = img[0]
                    
                    # --- FILTRO DE VELOCIDAD ---
                    # Obtenemos dimensiones sin extraer la imagen completa aún
                    base_image_info = doc.extract_image(xref)
                    width = base_image_info["width"]
                    height = base_image_info["height"]
                    image_bytes = base_image_info["image"]
                    
                    # 🚀 TRUCO: Si la imagen es muy chica (< 150px), es un logo o firma. IGNORAR.
                    if width < 150 or height < 150:
                        continue
                        
                    # Si es grande, procesamos con OCR
                    try:
                        resultado_ocr, _ = self.ocr_engine(image_bytes)
                        if resultado_ocr:
                            for linea in resultado_ocr:
                                if linea[1]:
                                    texto_ocr += linea[1] + " "
                    except:
                        continue # Si falla una imagen, seguimos con la siguiente
            
            # 3. Consolidar
            contenido_final = f"\n--- Página {numero_pagina + 1} ---\n"
            contenido_final += texto_pagina + "\n"
            if texto_ocr:
                contenido_final += f"[CONTENIDO VISUAL DETECTADO]:\n{texto_ocr}\n"
            
            texto_completo += contenido_final
            
            # Barra de progreso visual simple
            if (numero_pagina + 1) % 20 == 0:
                print(f"      ⚡ Progreso: {numero_pagina + 1}/{total_paginas} páginas procesadas...")
                
        return texto_completo

    def cargar_y_procesar(self, ruta_carpeta="datos_essalud"):
        inicio = time.time()
        if not os.path.exists(ruta_carpeta):
            print(f"❌ ERROR: No existe la carpeta '{ruta_carpeta}'")
            return []

        # Aumentamos chunk size para documentos legales grandes
        divisor = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""]
        )
        
        documentos_procesados = []
        archivos_pdf = [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf')]
        
        print(f"📚 Encontrados {len(archivos_pdf)} archivos PDF.")
        
        for archivo in archivos_pdf:
            ruta_completa = os.path.join(ruta_carpeta, archivo)
            print(f"🚀 Procesando: {archivo}")
            
            try:
                texto_total = self._extraer_texto_con_ocr(ruta_completa)
                
                if not texto_total.strip():
                    print("⚠️  Advertencia: Archivo vacío.")
                    continue
                
                fragmentos = divisor.split_text(texto_total)
                documentos_procesados.extend(fragmentos)
                
                tipo_doc = self._clasificar_tipo(archivo)
                for i, frag in enumerate(fragmentos):
                    self.metadatos.append({
                        'archivo': archivo,
                        'fragmento_id': i,
                        'tipo_documento': tipo_doc
                    })
                
                print(f"   ✅ Finalizado: {len(fragmentos)} fragmentos generados.")
                
            except Exception as e:
                print(f"   ❌ Error en {archivo}: {e}")

        self.fragmentos_documentos = documentos_procesados
        print(f"⏱️ Tiempo total de lectura: {time.time() - inicio:.2f} segundos")
        return documentos_procesados

    def _clasificar_tipo(self, nombre):
        n = nombre.lower()
        if 'seguridad' in n: return 'Directiva Seguridad Paciente'
        return 'Documento Normativo'

    def generar_embeddings_y_guardar(self, ruta_guardado="base_vectorial"):
        if not self.fragmentos_documentos:
            print("⚠️ No hay datos para guardar.")
            return

        print("\n🧠 VECTORIZANDO (Creando Inteligencia)...")
        embeddings = self.modelo.encode(
            self.fragmentos_documentos, 
            show_progress_bar=True,
            batch_size=32
        ).astype('float32')
        
        faiss.normalize_L2(embeddings)
        self.indice = faiss.IndexFlatIP(embeddings.shape[1])
        self.indice.add(embeddings)
        
        if not os.path.exists(ruta_guardado): os.makedirs(ruta_guardado)
        
        faiss.write_index(self.indice, os.path.join(ruta_guardado, "indice_vectorial.faiss"))
        with open(os.path.join(ruta_guardado, "fragmentos.pkl"), 'wb') as f:
            pickle.dump(self.fragmentos_documentos, f)
        with open(os.path.join(ruta_guardado, "metadatos.pkl"), 'wb') as f:
            pickle.dump(self.metadatos, f)
            
        print(f"💾 BASE DE DATOS GUARDADA EN: {ruta_guardado}/")

if __name__ == "__main__":
    base = BaseVectorialEsSalud()
    base.cargar_y_procesar()
    base.generar_embeddings_y_guardar()