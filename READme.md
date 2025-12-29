# 🏥 Asistente Virtual RAG - EsSalud AI

Sistema de Inteligencia Artificial capaz de responder preguntas sobre normativas de seguridad del paciente, directivas y trámites de EsSalud (Seguro Social del Perú), utilizando documentos oficiales (PDFs).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?style=for-the-badge&logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange?style=for-the-badge&logo=google&logoColor=white)

## 🚀 Características

- **RAG (Retrieval-Augmented Generation):** Búsqueda semántica precisa en documentos PDF.
- **Visión Artificial (OCR):** Lee tablas, gráficos e infografías dentro de los PDFs que otros sistemas ignoran.
- **Modelos Híbridos:** Soporte para Google Gemini (Flash 1.5/2.0) y OpenAI.
- **Interfaz Moderna:** Chat web con diseño "Glassmorphism" premium.
- **Búsqueda Inteligente:** Expande siglas automáticamente (ej: "CITT" -> "Certificado de Incapacidad...").

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone [https://github.com/aritoo07/BaseVectorial_essalud.git](https://github.com/aritoo07/BaseVectorial_essalud.git)
cd BaseVectorial_essalud

### 2.Crea un entorno virtual e instala las dependencias

python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.


### 3-Configurar variables de entorno
Crea archivo .env

GEMINI_API_KEY=tu_clave_de_google_aqui
OPENAI_API_KEY=tu_clave_de_openai_aqui (opcional)

### 4.Configuración del Modelo
GEMINI_MODEL=gemini-1.5-flash
DEFAULT_LLM=gemini
VECTOR_DB_PATH=base_vectorial

### 5.Construir la base de conocimientos
python src/constructor_vectorial.py

### 6.Ejecutar la API
python -m uvicorn app.main:app --reload