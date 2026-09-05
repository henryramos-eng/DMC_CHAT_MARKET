"""Adaptadores pequenos para embeddings y generacion con OpenAI."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("Falta OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(
        self, texts: Sequence[str], *, batch_size: int = 64
    ) -> np.ndarray:
        if not texts:
            raise ValueError("No hay textos para generar embeddings")
        rows: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )
            rows.extend(item.embedding for item in response.data)
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class OpenAIAnswerGenerator:
    INSTRUCTIONS = """Eres un asistente para usuarios internos del mercado de granos.
Responde siempre en espanol, salvo que el usuario pida expresamente el texto original en ingles.
Usa exclusivamente el contexto suministrado de los reportes StoneX.
No uses conocimiento general para completar vacios y no inventes cifras, fechas, causas o proyecciones.
Distingue hechos reportados, expectativas y comentarios editoriales cuando el contexto lo permita.
Si el contexto no permite responder, devuelve exactamente: NO_ENCONTRADO
Consolida fragmentos solapados y evita repetir la misma informacion.
No inventes citas ni paginas; la aplicacion agregara las fuentes de forma determinista."""

    STYLE_INSTRUCTIONS = {
        "answer": (
            "Responde de forma directa y profesional, normalmente en uno a tres "
            "parrafos. Prioriza lo solicitado por el usuario."
        ),
        "summary": (
            "Genera un resumen ejecutivo del reporte completo. Abre con la idea "
            "principal y continua con viñetas breves por tema. Incluye solamente "
            "los temas presentes en el contexto."
        ),
        "bulletin": (
            "Genera un boletin ordenado en Markdown. Incluye un titulo con la fecha, "
            "un panorama general y secciones tematicas como soja, maiz, trigo, clima, "
            "exportaciones o fondos solo cuando exista evidencia. Separa hechos de "
            "expectativas y omite cualquier seccion sin informacion."
        ),
    }

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("Falta OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(
        self,
        question: str,
        context: str,
        *,
        response_style: str = "answer",
    ) -> str:
        style = self.STYLE_INSTRUCTIONS.get(
            response_style,
            self.STYLE_INSTRUCTIONS["answer"],
        )
        prompt = (
            "CONTEXTO RECUPERADO DE LOS PDF:\n"
            f"{context}\n\n"
            "FORMATO SOLICITADO:\n"
            f"{style}\n\n"
            "PREGUNTA DEL USUARIO:\n"
            f"{question}"
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=self.INSTRUCTIONS,
            input=prompt,
            store=False,
        )
        return response.output_text.strip()
