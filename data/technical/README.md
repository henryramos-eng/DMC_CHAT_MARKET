# Fuente técnica

`checklist_tecnico.xlsx` es una fuente independiente del RAG.

El comando `technical_ingest.py` valida el archivo y genera
`data/processed/technical/checklist_tecnico.json`. El Excel y el JSON nunca se
incorporan al índice vectorial ni generan embeddings.

La estructura actual contiene `Checklist_Resumen` y `Checklist_Detalle`. El
adaptador extrae cierre, EMA50, EMA200, RSI y MACD; calcula variación porcentual,
MA20 y MA50 usando únicamente historia disponible hasta cada fecha.

El archivo no contiene soporte ni resistencia, por lo que esos campos se
mantienen como `null` y se muestran como `n.d.` en el boletín.
