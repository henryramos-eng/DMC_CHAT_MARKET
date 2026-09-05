# CHAT_MARKET_TEST

Chatbot RAG para consultar en español una colección de reportes diarios en
inglés `morning_grain_comments_DDMMYYYY.pdf` mediante Telegram.

La recuperación combina similitud semántica con fechas de publicación. Si el
usuario no indica una fecha, el bot busca el tema en toda la colección y ofrece
únicamente fechas con coincidencias suficientes. No utiliza LangChain, agentes,
ChromaDB ni SQL.

## Comportamiento conversacional

La bienvenida rota grupos de ejemplos para mostrar diferentes capacidades:
consultas temáticas, clima, países, tendencias, resúmenes, boletines y listado
de reportes.

Ejemplos admitidos:

- `Dame novedades sobre China`
- `clima en Argentina 03092026`
- `Resumen al día 04092026`
- `Quiero un boletín ordenado del 03/09/2026`
- `tendencia actual del mercado`
- `qué fechas hay`

Cuando una consulta sin fecha sí está relacionada con el contenido, Telegram
muestra opciones. El usuario puede pulsar un botón o responder con el número o
la fecha. Una consulta de seguimiento como `¿y Argentina?` conserva la última
fecha utilizada.

Si la pregunta es ajena a los reportes o no supera el umbral de relevancia, el
bot explica su alcance y no muestra fechas. Una fecha inexistente solo produce
alternativas cuando hay evidencia temática real en otros reportes.

## Formatos de salida

- Consulta normal: respuesta directa sobre los chunks más relevantes.
- Resumen: utiliza todos los chunks del reporte elegido y produce una síntesis.
- Boletín: utiliza el reporte completo y lo organiza por temas, omitiendo
  secciones sin evidencia.

El FAQ solamente evalúa el sistema; no limita las preguntas permitidas y nunca
se incorpora al índice vectorial.

## Agregar reportes

Guarda cada archivo en `data/raw` con el formato:

```text
morning_grain_comments_DDMMYYYY.pdf
```

Después ejecuta:

```powershell
.\.venv\Scripts\python.exe .\rag_index.py --force
.\.venv\Scripts\python.exe .\rag_bot.py
```

Para cambios exclusivamente conversacionales basta reiniciar el bot.

No compartas ni confirmes `.env` en Git.
