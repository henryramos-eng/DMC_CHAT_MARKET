# Fase 3: generación editorial con OpenAI

## Flujo

```text
RAG en modo lectura + snapshot técnico del Excel
→ OpenAI Responses API con Structured Outputs
→ validación de cinco commodities
→ guardrails de evidencia
→ cinco PNG
→ Telegram
```

El backend predeterminado es `openai`. Reutiliza `OPENAI_API_KEY` y, si no se
define `BULLETIN_OPENAI_MODEL`, reutiliza `OPENAI_LLM_MODEL`.

## Controles

- La respuesta usa JSON Schema estricto.
- Debe incluir exactamente QBS, QSM, QBO, CL y HO.
- Cada commodity contiene dos líneas de análisis y tres factores.
- Los textos vacíos se reemplazan por una declaración explícita de falta de
  evidencia, nunca por información inventada.
- Clasificación y análisis técnico se calculan mediante reglas sobre el Excel.
- Los factores documentales del LLM se utilizan en QBS, QSM y QBO.
- CL y HO utilizan factores técnicos porque el PDF documental actual es de
  granos y no aporta cobertura energética suficiente.
- Si OpenAI falla, el pipeline usa la plantilla determinista y registra
  `template_fallback:<Error>` en el manifiesto.
- Las llamadas usan `store=False`.

## Variables

```dotenv
BULLETIN_TEXT_BACKEND=openai
BULLETIN_OPENAI_MODEL=
BULLETIN_OPENAI_TIMEOUT_SECONDS=60
BULLETIN_LLM_FALLBACK=true
```

## Trazabilidad

Cada ejecución guarda:

- `context.json`: evidencia suministrada al pipeline;
- `draft.json`: contenido estructurado y backend utilizado;
- `manifest.json`: backend, guardrails, número y rutas de las imágenes;
- cinco PNG verticales.

LoRA permanece desactivado hasta contar con boletines históricos revisados y
marcados como validados.
