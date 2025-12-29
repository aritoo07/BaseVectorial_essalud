from google import genai
from google.genai import types
from openai import OpenAI
from typing import List, Dict, Any, Optional
import time
import logging
import os
import random
from app.config import settings

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMService")

class LLMService:
    def __init__(self):
        # --- Configuración OpenAI ---
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI configurado.")
            except Exception as e:
                logger.warning(f"⚠️ Error OpenAI: {e}")

        # --- Configuración Gemini (NUEVO SDK) ---
        self.gemini_client = None
        self.gemini_available = False
        
        if settings.GEMINI_API_KEY:
            try:
                # 🛑 IMPORTANTE: Forzamos 'v1beta' para acceder a modelos nuevos (2.0)
                self.gemini_client = genai.Client(
                    api_key=settings.GEMINI_API_KEY,
                    http_options={'api_version': 'v1beta'} 
                )
                self.gemini_model_name = settings.GEMINI_MODEL
                self.gemini_available = True
                logger.info(f"✅ Gemini configurado (Nuevo SDK / v1beta): {self.gemini_model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Error Gemini: {e}")
        
        self.default_provider = settings.DEFAULT_LLM

    def _format_history(self, history: List[Dict]) -> str:
        if not history: return "Sin historial."
        formatted = []
        for msg in history[-6:]: 
            role = "Usuario" if msg['role'] == 'user' else "Asistente"
            content = str(msg.get('content', '')).replace('\n', ' ')
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        if not documents: return "No se encontraron documentos relevantes."
        parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.get('metadata', {})
            parts.append(f"🔴 DOCUMENTO {i} [Fuente: {meta.get('archivo', '?')}] [Tipo: {meta.get('tipo_documento', 'General')}]\nCONTENIDO:\n{doc.get('content', '')}\n")
        return "\n".join(parts)

    def _call_gemini_with_retry(self, prompt: str, max_tokens: int, retries=3) -> str:
        """Llama a Gemini usando el NUEVO SDK 'google-genai'."""
        if not self.gemini_available or not self.gemini_client:
            return "Error: Cliente Gemini no inicializado."
        
        for attempt in range(retries):
            try:
                # Sintaxis nueva: client.models.generate_content
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.3
                    )
                )
                return response.text if response.text else "El modelo no generó respuesta."
            
            except Exception as e:
                error_str = str(e).lower()
                # Manejo de error 429 (Cuota) o Resource Exhausted
                if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    if attempt < retries - 1:
                        wait_time = (attempt + 1) * 5 
                        logger.warning(f"⚠️ Cuota Gemini excedida. Reintentando en {wait_time}s... (Intento {attempt+1}/{retries})")
                        time.sleep(wait_time)
                        continue
                
                logger.error(f"❌ Error Gemini (SDK Nuevo): {e}")
                return "Lo siento, hubo un problema técnico con el servicio de IA de Google."

    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        if not self.openai_client: return "Error: OpenAI no configurado."
        try:
            resp = self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Error OpenAI API: {e}")
            return "Lo siento, hubo un error técnico con OpenAI."

    # --- Lógica Principal ---

    def contextualize_query(self, query: str, history: List[Dict]) -> str:
        history_text = self._format_history(history)
        
        prompt = (
            "Eres un experto en terminología de EsSalud. Tu trabajo es PREPARAR la consulta para búsqueda.\n"
            "REGLAS OBLIGATORIAS: \n"
            "1. Expande TODAS las siglas (CITT -> Certificado de Incapacidad..., CAS -> Contrato Admin...).\n"
            "2. Corrige ortografía y gramática.\n"
            "3. Si la pregunta depende del historial, complétala.\n"
            "SOLO devuelve la frase reescrita sin explicaciones.\n\n"
            f"Historial: {history_text}\n"
            f"Consulta Original: {query}\n"
            "Consulta Optimizada:"
        )

        try:
            if self.default_provider == 'gemini' and self.gemini_available:
                return self._call_gemini_with_retry(prompt, 200, retries=1).strip().replace('"', '')
            elif self.openai_client:
                return self._call_openai(prompt, 200).strip().replace('"', '')
            return query
        except:
            return query

    def generate_answer(self, query: str, context_documents: List[Dict], history: List[Dict], max_tokens: int = 1000, provider: str = None) -> str:
        context = self._build_context(context_documents)
        history_text = self._format_history(history)
        
        system_prompt = (
            "Eres el Asistente Oficial de EsSalud. Tu objetivo es informar basándote EXCLUSIVAMENTE en la documentación adjunta.\n\n"
            
            "### 🛡️ REGLAS DE RESPUESTA:\n"
            "1. **VERACIDAD:** Si la respuesta no está en el contexto, di 'No tengo esa información en los documentos cargados'.\n"
            "2. **FORMATO DE DATOS:** El contexto puede contener tablas Markdown y texto extraído de imágenes. Úsalos como fuentes válidas.\n"
            "3. **ESTILO DE REDACCIÓN:**\n"
            "   - Escribe de forma fluida y profesional.\n"
            "   - **NO repitas la fuente en cada línea**.\n"
            "   - Menciona el documento fuente al inicio o final del bloque.\n"
            "   - Usa **negritas** para conceptos clave.\n"
            "4. **ESTRUCTURA:** Usa listas (bullets) limpias.\n\n"
            
            f"### 📚 DOCUMENTACIÓN RECUPERADA:\n{context}\n\n"
            f"### 💬 HISTORIAL:\n{history_text}\n\n"
            f"### ❓ PREGUNTA:\n{query}\n\n"
            "Respuesta:"
        )

        chosen_provider = (provider or self.default_provider).lower()
        
        if chosen_provider == 'gemini':
            return self._call_gemini_with_retry(system_prompt, max_tokens)
        elif chosen_provider == 'openai':
            return self._call_openai(system_prompt, max_tokens)
        else:
            return "Proveedor no válido."