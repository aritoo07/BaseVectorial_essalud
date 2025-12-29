# 🏥 Asistente Virtual RAG - EsSalud AI

Sistema de Inteligencia Artificial capaz de responder preguntas sobre normativas de seguridad del paciente, directivas y trámites de EsSalud (Seguro Social del Perú), utilizando documentos oficiales (PDFs).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange)

## 🚀 Características
- **RAG (Retrieval-Augmented Generation):** Búsqueda semántica en documentos PDF.
- **Visión Artificial (OCR):** Lee tablas e infografías dentro de los PDFs.
- **Modelos Híbridos:** Soporte para Google Gemini (Flash 1.5/2.0) y OpenAI.
- **Interfaz Moderna:** Chat web con diseño "Glassmorphism".
- **Búsqueda Inteligente:** Expande siglas automáticamente (ej: "CITT" -> "Certificado de Incapacidad...").

## 🛠️ Instalación

1. Clonar el repositorio:
   ```bash
   git clone <URL_DE_TU_REPO>
   cd essalud_rag_api


# Crear entorno virtual e instalar dependencias:

# python -m venv .venv
#       .venv\Scripts\activate
#   pip install -r requirements.


# ------------------------------------------------------
# Configurar variables de entorno: Crea un archivo .env y agrega tus claves:

# Fragmento de código

 GEMINI_API_KEY=tu_clave_aqui
 OPENAI_API_KEY=tu_clave_aqui
 GEMINI_MODEL=gemini-1.5-flash
 DEFAULT_LLM=gemini
 VECTOR_DB_PATH=base_vectorial 

## Construir la base de conocimientos (Leer PDFs):

python src/constructor_vectorial.py 

# Ejecutar la API

python -m uvicorn app.main:app --reload
