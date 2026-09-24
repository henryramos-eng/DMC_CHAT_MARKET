# Pipeline de boletines en imagen

## Separación del RAG

El índice RAG existente se abre en modo lectura mediante `CurrentRAGAdapter`.
No se extraen nuevamente los PDF, no se recalculan chunks y no se generan
embeddings. El Excel técnico se procesa por un flujo separado y se convierte a
JSON antes de construir el contexto unificado.

```text
Índice RAG existente ─┐
                     ├─ contexto unificado ─ generador ─ renderer ─ PNG
Excel ─ JSON técnico ┘
```

## Excel técnico

```powershell
.\.venv\Scripts\python.exe .\technical_ingest.py
```

Validaciones:

- firma y lectura XLSX;
- hojas y columnas obligatorias;
- fechas y campos requeridos;
- unicidad de `Commodity + Fecha`;
- unicidad de `Commodity + Fecha + Indicador`;
- interpretación completa de Close, EMA50, EMA200, RSI y MACD;
- dominio controlado de tendencias.

El archivo actual no contiene soporte ni resistencia. Ambos se exportan como
`null`. MA20 y MA50 se calculan con el historial de cierres sin utilizar datos
posteriores a la fecha evaluada.

## Generación local

```powershell
.\.venv\Scripts\python.exe .\bulletin_generate.py --date 04092026
```

Se generan cuatro páginas PNG:

1. resumen ejecutivo;
2. precios, MA20 y MA50;
3. RSI;
4. tablero técnico.

Los archivos quedan en `data/generated/bulletins/YYYY-MM-DD/` junto con el
contexto unificado, el borrador y un manifiesto de trazabilidad.

## Telegram

```text
/boletin
/boletin 04092026
```

Sin fecha se utiliza la fecha más reciente disponible simultáneamente en RAG y
Excel. Telegram recibe las cuatro páginas como álbum.

## LoRA

La Fase 1 utiliza `BULLETIN_TEXT_BACKEND=template` para probar el flujo sin
presentar ejemplos sintéticos como un modelo entrenado. El backend LoRA queda
preparado, pero no debe activarse hasta contar con:

- un modelo base open-weight compatible;
- hardware validado;
- suficientes boletines históricos aprobados;
- evaluación separada por fecha.

Los ejemplos de `training/data/examples.jsonl` son sintéticos y solo sirven
para verificar la preparación del dataset:

```powershell
.\.venv\Scripts\python.exe .\training\prepare_dataset.py --allow-synthetic
```

No se ejecuta `train_lora.py` automáticamente.
