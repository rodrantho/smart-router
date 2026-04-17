# Smart Router

**Proxy inteligente OpenAI-compatible para Ollama.** Clasifica cada query por complejidad y la enruta al modelo local apropiado — respuestas rápidas con modelos chicos, razonamiento con modelos medianos, código con los modelos grandes. Todo corre local, gratis.

Incluye un **dashboard web** en tiempo real con estadísticas de uso, ahorro estimado vs la nube y un chat de prueba.

![Smart Router Dashboard](docs/dashboard.png)

---

## ¿Por qué?

Tener un solo LLM grande siempre cargado desperdicia VRAM y tiempo. Un "hola" no necesita llama3.1:8b. Este proxy:

- **Clasifica** la query con un modelo minúsculo (gemma3:1b, <1s)
- **Rutea** al modelo más chico que pueda con la tarea
- **Expone** una API compatible con OpenAI (`/v1/chat/completions`), así cualquier cliente (OpenWebUI, LibreChat, Continue, Cline, LobeChat, etc.) lo usa sin cambios
- **Tracking** — historial, tokens, tiempos, ahorro estimado vs GPT-4o-mini / o1-mini / Claude Sonnet

---

## Cómo rutea

| Nivel | Cuándo se usa | Modelo local | Equivale en nube |
|---|---|---|---|
| `rapido` | saludos, chitchat, respuestas cortas | `qwen2.5:1.5b` | GPT-4o-mini |
| `medio` | conocimiento general, definiciones, explicaciones simples | `gemma4:e2b` | GPT-4o-mini |
| `razonamiento` | matemática, lógica, análisis, decisiones | `phi4-mini-reasoning` | o1-mini |
| `complejo` | código, debugging, tareas multi-paso | `llama3.1:8b` | Claude Sonnet |

El mapeo se edita fácil en `smart_router.py` — usá los modelos que tengas.

---

## Requisitos previos

1. **Python 3.10+**
2. **Ollama** instalado y corriendo → https://ollama.com/download
3. **Los modelos descargados**:

   ```bash
   ollama pull gemma3:1b              # clasificador
   ollama pull qwen2.5:1.5b           # rapido
   ollama pull llama3.1:8b            # complejo
   ollama pull phi4-mini-reasoning    # razonamiento
   # 'medio' usa gemma4:e2b — si no lo tenés, cambialo por gemma3:4b o similar
   ```

4. **GPU con ~6 GB VRAM** recomendado (el más pesado es llama3.1:8b ≈ 4.7 GB Q4). En CPU también funciona pero lento.

### Tips de Ollama

- **Unload rápido de VRAM** (importante si vas a intercalar modelos):
  ```bash
  # Linux/Mac
  export OLLAMA_KEEP_ALIVE=0
  # Windows (PowerShell)
  setx OLLAMA_KEEP_ALIVE 0
  ```
  Después reiniciá Ollama.

- **Exponer Ollama a la red** (opcional, si el router corre en otra máquina que Ollama):
  ```bash
  export OLLAMA_HOST=0.0.0.0
  ollama serve
  ```

- **Ampliar contexto** — si necesitás ventanas más grandes, creá un `Modelfile`:
  ```
  FROM llama3.1:8b
  PARAMETER num_ctx 32768
  ```
  Y corré: `ollama create llama3.1-32k -f Modelfile`

---

## Instalación

```bash
git clone https://github.com/TU_USUARIO/smart-router.git
cd smart-router
pip install -r requirements.txt
python smart_router.py
```

El router queda escuchando en `http://0.0.0.0:6061`.

**Dashboard web:** http://localhost:6061
**API docs (Swagger):** http://localhost:6061/docs

### Instalación en Windows — nota de encoding

Si ves `UnicodeEncodeError` en consola, arrancá con UTF-8 explícito:

```bash
PYTHONUTF8=1 python smart_router.py
```

---

## Uso

### Como API (compatible OpenAI)

```bash
curl -X POST http://localhost:6061/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart-router",
    "messages": [{"role": "user", "content": "escribime un script python para parsear un csv"}],
    "stream": false
  }'
```

El campo `"model"` acepta:

- `smart-router` — que decida el router
- `rapido` / `medio` / `razonamiento` / `complejo` — forzás un nivel
- cualquier nombre de modelo de Ollama — pasa directo sin clasificar

### Como backend de clientes existentes

Configurá cualquier cliente OpenAI-compatible con:

- **Base URL:** `http://localhost:6061/v1`
- **API Key:** cualquier string (no se valida)
- **Model:** `smart-router`

Funciona con **OpenWebUI, LibreChat, Continue, Cline, Cursor, LobeChat, Open Interpreter**, etc.

### Streaming

```bash
curl -N -X POST http://localhost:6061/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smart-router","messages":[{"role":"user","content":"hola"}],"stream":true}'
```

---

## Dashboard

En `http://localhost:6061` encontrás:

- **Contadores en vivo** — total requests, ahorro estimado, tokens, modelo más usado
- **Donut chart** de distribución por nivel
- **Chat de prueba** — escribí y mirá qué modelo eligió el router
- **Historial** de las últimas 200 requests con tokens, tiempo y costo estimado
- **Grid de modelos** con métricas por nivel

El ahorro se calcula asumiendo pricing del equivalente cloud (tabla `CLOUD_EQUIV` en `smart_router.py`, editable).

---

## Personalización

### Cambiar modelos

Editá el diccionario `MODELS` arriba de `smart_router.py`:

```python
MODELS = {
    "rapido":       "qwen2.5:1.5b",
    "medio":        "gemma4:e2b",
    "razonamiento": "phi4-mini-reasoning:latest",
    "complejo":     "llama3.1:8b",
}
```

### Cambiar el clasificador

Por defecto usa `gemma3:1b` (rápido). Podés cambiarlo por cualquier modelo chico:

```python
CLASSIFIER_MODEL = "gemma3:1b"
```

### Editar las categorías

Tocá `CLASSIFY_PROMPT` — son few-shot examples. Podés cambiar los labels o agregar más niveles (acordate de actualizar `MODELS` y `CLOUD_EQUIV`).

### Cambiar el puerto

Al final del archivo:

```python
uvicorn.run(app, host="0.0.0.0", port=6061)
```

---

## Cómo funciona

1. Llega un POST a `/v1/chat/completions`
2. Si `model == "smart-router"`, extrae el último mensaje del usuario
3. Lo manda al clasificador (`gemma3:1b`) con un prompt few-shot pidiendo **una sola palabra**
4. Sanitiza la respuesta (solo ASCII, lowercase), busca coincidencias con `rapido/medio/razonamiento/complejo`
5. Si hay match → rutea al modelo mapeado. Si no → cae a `medio` por defecto
6. Proxy-ea la request a Ollama (`/api/chat`), streaming opcional
7. Guarda en el historial: query, nivel, modelo, tokens, tiempo, costo estimado

---

## FAQ

**¿Necesito internet?** No. Ollama corre 100% local. Los precios de cloud son tabla hardcodeada.

**¿El clasificador siempre acierta?** No, es un modelo chico. En dudas cae a `medio`. Si te importa la precisión, usá un clasificador más grande (`gemma3:4b`) o uno fine-tuneado.

**¿Se puede usar un modelo cloud (OpenAI, Claude) en algún nivel?** Sí — reemplazá el flujo de Ollama por un cliente HTTP al proveedor que quieras. El dispatcher ya queda listo.

**¿Funciona en Mac / Linux?** Sí. Probado en Windows 11 + WSL2, macOS, Linux.

**¿Por qué mis requests son lentas al inicio?** `OLLAMA_KEEP_ALIVE=0` descarga modelos entre llamadas. La primera request de un modelo recién descargado tarda en cargarse a VRAM. Si priorizás velocidad, subí el keep-alive (ej. `5m`).

---

## Roadmap

- [ ] Clasificador con embeddings en vez de LLM (más rápido)
- [ ] Cache de clasificaciones por hash de query
- [ ] Soporte nativo para function calling
- [ ] Export de métricas a Prometheus
- [ ] WebSocket para updates del dashboard en tiempo real

---

## Licencia

MIT — usá, modificá y distribuí libremente.

## Contribuir

PRs bienvenidos. Issues también. Si tenés un caso de uso raro o querés agregar soporte para otro proveedor (llama.cpp, LM Studio, vLLM), abrí un issue primero para discutir el approach.
