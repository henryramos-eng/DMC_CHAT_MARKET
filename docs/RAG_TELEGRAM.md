# Guía conversacional de CHAT_MARKET_TEST

## Compuerta temática

Las fechas solo se ofrecen cuando la consulta obtiene coincidencias semánticas
que superan el umbral configurado. Si no existe evidencia, el bot responde que
la información no está en los reportes y explica brevemente su alcance.

Las preguntas evidentemente ajenas, como consultas sobre vehículos, fútbol,
recetas o programación, se rechazan localmente antes de generar un embedding.
La búsqueda semántica actúa como segunda barrera para casos no previstos por
esas reglas. En ninguno de ambos casos se muestran fechas.

## Selección dinámica de fecha

Ante una pregunta como `Dame novedades sobre China`, el sistema crea un
embedding, busca en la colección y agrupa las coincidencias por fecha. Telegram
presenta hasta cinco opciones mediante botones. También se puede responder con
`1`, `2`, una fecha o `cancelar`.

Una fecha inexistente solo ofrece alternativas cuando el mismo tema tiene
evidencia suficiente en otros reportes. Nunca se presenta una alternativa como
si fuera la fecha solicitada.

## Bienvenida variable

Cada saludo alterna un conjunto de tres preguntas sugeridas. Las sugerencias
abarcan búsquedas abiertas, países, clima, materias primas, reportes disponibles,
resúmenes y boletines.

## Resúmenes y boletines

`Resumen al día 04092026` utiliza todos los chunks del documento de esa fecha.
`Boletín ordenado 04092026` usa el mismo contexto completo y solicita una salida
en Markdown por temas, omitiendo secciones sin evidencia.

## Operación

Estos cambios no modifican los embeddings. Basta reiniciar el bot:

```powershell
.\.venv\Scripts\python.exe .\rag_bot.py
```

Solo ejecuta `rag_index.py --force` cuando cambies los PDF, el chunking o el
modelo de embeddings.
