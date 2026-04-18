# -*- coding: utf-8 -*-
# Smart Router - routes queries to the right model based on complexity
import sys
import httpx
import json
import time
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = "http://localhost:11434"

MODELS = {
    "rapido":       "qwen2.5:1.5b",              # ~1 GB   40 tok/s — chat simple
    "medio":        "qwen2.5:3b",                 # ~1.9 GB 21 tok/s — conocimiento general
    "razonamiento": "phi4-mini-reasoning:latest", # ~3.2 GB          — math, lógica
    "complejo":     "gemma4:e2b",                 # ~3.1 GB          — código, tools, agentes
}

# Equivalent cloud pricing USD per 1M tokens (input / output)
# These are the models you would have used in the cloud for the same task
CLOUD_EQUIV = {
    "rapido":       {"name": "GPT-4o-mini",   "in": 0.15,  "out": 0.60},
    "medio":        {"name": "GPT-4o",        "in": 2.50,  "out": 10.00},
    "razonamiento": {"name": "o1-mini",       "in": 1.10,  "out": 4.40},
    "complejo":     {"name": "GPT-4o",        "in": 2.50,  "out": 10.00},
}

CLASSIFIER_MODEL = "gemma3:1b"

CLASSIFY_PROMPT = (
    'You are a query classifier. Read the user query and reply with EXACTLY one word.\n\n'
    'Categories:\n'
    '- rapido: hello, hi, thanks, bye, simple chitchat, one-word answers\n'
    '- medio: general knowledge, definitions, simple explanations, history, facts\n'
    '- razonamiento: math problems, logic puzzles, analysis, reasoning, comparisons\n'
    '- complejo: code, programming, scripts, debugging, multi-step tasks, file operations\n\n'
    'Examples:\n'
    'Query: "hello" -> rapido\n'
    'Query: "hola" -> rapido\n'
    'Query: "what is photosynthesis" -> medio\n'
    'Query: "write a python script" -> complejo\n'
    'Query: "solve 2+2" -> razonamiento\n'
    'Query: "explain recursion with code" -> complejo\n\n'
    'Query: "{query}"\n\n'
    'Reply with ONE word only (rapido/medio/razonamiento/complejo):'
)

history = deque(maxlen=200)
stats = {k: {"count": 0, "tokens_in": 0, "tokens_out": 0, "cost_saved": 0.0, "ms_total": 0} for k in MODELS}
start_time = time.time()

app = FastAPI(title="Smart Router", docs_url="/docs")


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Tool-call format converters ──────────────────────────────────────────────

def tc_ollama_to_openai(tool_calls: list) -> list:
    """Convert Ollama tool_call list → OpenAI format expected by clients."""
    result = []
    for i, tc in enumerate(tool_calls or []):
        func = tc.get("function") or {}
        args = func.get("arguments", {})
        # Ollama returns arguments as a dict; OpenAI wants a JSON string
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        result.append({
            "id": tc.get("id") or f"call_{i:06x}",
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "arguments": args,
            },
        })
    return result


def tc_openai_to_ollama(tool_calls: list) -> list:
    """Convert OpenAI tool_call list → Ollama format for forwarding."""
    result = []
    for tc in tool_calls or []:
        func = (tc.get("function") or {}).copy()
        args = func.get("arguments", {})
        # OpenAI arguments is a JSON string; Ollama wants a dict
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        func["arguments"] = args
        result.append({"function": func})
    return result


def normalize_messages_for_ollama(messages: list) -> list:
    """
    Flatten message list for Ollama:
    - content: list-of-blocks → plain string
    - assistant tool_calls: OpenAI format → Ollama format
    - role=tool messages: drop tool_call_id (Ollama doesn't use it)
    - preserve content=None as "" (Ollama needs a string)
    """
    norm = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        nm = {k: v for k, v in m.items()}          # shallow copy
        nm["content"] = msg_text(m) or ""           # always string
        # convert assistant tool_calls to Ollama format
        if nm.get("tool_calls"):
            nm["tool_calls"] = tc_openai_to_ollama(nm["tool_calls"])
        # Ollama doesn't use tool_call_id on role=tool messages
        nm.pop("tool_call_id", None)
        # Ollama doesn't understand name field on tool messages
        if nm.get("role") == "tool":
            nm.pop("name", None)
        norm.append(nm)
    return norm


def msg_text(msg) -> str:
    """Extrae el texto de un mensaje OpenAI. `content` puede ser str o lista
    de bloques (formato multimodal / tools)."""
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(c) if c is not None else ""


def calc_cost(level: str, tokens_in: int, tokens_out: int) -> float:
    """Cost this would have been on the cloud equivalent"""
    p = CLOUD_EQUIV.get(level)
    if not p:
        return 0.0
    return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]


# Fast heuristic classification - avoids a classifier call when the answer is obvious
CODE_MARKERS = ("```", "def ", "function ", "import ", "from ", "class ",
                "const ", " let ", " var ", "npm ", "pip install", "SELECT ",
                "SELECT\n", "error:", "traceback", "stacktrace", "stack trace",
                "syntaxerror", "typeerror", "<script", "<html", "curl ", "git ",
                "sudo ", "async ", "await ", "=>", " && ", " || ")

REASONING_MARKERS = ("solve", "calcula", "calculate", "compute", "demuestra",
                     "prove that", "step by step", "paso a paso", "why does",
                     "por que ", "how many", "cuantos ", "cuantas ")


def heuristic_classify(query: str) -> str | None:
    """Return a level if the query is obviously in one category, else None."""
    q = (query or "").strip()
    if not q:
        return "rapido"
    ql = q.lower()
    if len(q) < 40:
        return "rapido"
    if any(m in ql for m in CODE_MARKERS):
        return "complejo"
    if any(m in ql for m in REASONING_MARKERS):
        return "razonamiento"
    return None


async def classify(query: str) -> str:
    # Try heuristics first - saves ~1-2s per request
    h = heuristic_classify(query)
    if h:
        print(f"  [Router] heuristic match: {h} (len={len(query)})")
        return h
    short = query[:150].replace("\n", " ")
    print(f"  [Router] classifying (len={len(query)}): {short!r}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{OLLAMA}/api/generate",
                json={
                    "model": CLASSIFIER_MODEL,
                    "prompt": CLASSIFY_PROMPT.format(query=short),
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 5},
                },
            )
            response = r.json().get("response", "")
            response_clean = "".join(c for c in response if c.isascii()).strip().lower()
            print(f"  [Router] classifier raw={response!r} clean={response_clean!r}")
            for key in MODELS:
                if key in response_clean:
                    return key
    except Exception as e:
        import traceback
        print(f"  [Router] classify error: {type(e).__name__}: {e}")
        traceback.print_exc()
    return "medio"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/history")
async def get_history():
    totals = {
        "count": sum(s["count"] for s in stats.values()),
        "tokens_in": sum(s["tokens_in"] for s in stats.values()),
        "tokens_out": sum(s["tokens_out"] for s in stats.values()),
        "cost_saved": round(sum(s["cost_saved"] for s in stats.values()), 6),
        "uptime": int(time.time() - start_time),
    }
    avg_ms = {k: (s["ms_total"] // s["count"]) if s["count"] else 0 for k, s in stats.items()}
    return {
        "history": list(history),
        "stats": stats,
        "totals": totals,
        "avg_ms": avg_ms,
        "models": MODELS,
        "cloud_equiv": CLOUD_EQUIV,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "smart-router", "object": "model", "owned_by": "local"},
            *[{"id": name, "object": "model", "owned_by": "local"} for name in MODELS],
        ],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    requested = body.get("model", "smart-router")
    t_start = time.time()

    # Debug log when tools are present or conversation has tool turns
    has_tools = "tools" in body
    has_tool_msgs = any(m.get("role") == "tool" for m in messages)
    if has_tools or has_tool_msgs:
        print(f"  [Router] TOOL REQUEST: msgs={len(messages)} has_tools={has_tools} has_tool_msgs={has_tool_msgs}")
        for i, m in enumerate(messages[-4:]):   # last 4 msgs
            role = m.get("role", "?")
            content_preview = str(msg_text(m))[:80].replace('\n', ' ')
            tc = m.get("tool_calls")
            tc_names = [t.get("function", {}).get("name") for t in (tc or [])]
            print(f"    msg[-{min(4,len(messages))-i}] role={role} tc={tc_names} content={content_preview!r}")

    if requested in MODELS:
        model = MODELS[requested]
        level = requested
    elif "router" in requested or requested == "smart-router":
        last_user = next(
            (msg_text(m) for m in reversed(messages) if m.get("role") == "user"), ""
        )
        level = await classify(last_user)
        model = MODELS[level]
    else:
        model = requested
        level = "medio"

    print(f"  [Router] level={level} -> model={model} (msgs={len(messages)})")

    last_user_msg = next(
        (msg_text(m) for m in reversed(messages) if m.get("role") == "user"), ""
    )
    prompt_text = " ".join(msg_text(m) for m in messages)
    tokens_in = est_tokens(prompt_text)

    def record(content: str, elapsed: float):
        tokens_out = est_tokens(content)
        cost = calc_cost(level, tokens_in, tokens_out)
        if level in stats:
            stats[level]["count"] += 1
            stats[level]["tokens_in"] += tokens_in
            stats[level]["tokens_out"] += tokens_out
            stats[level]["cost_saved"] += cost
            stats[level]["ms_total"] += int(elapsed * 1000)
        history.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "query": last_user_msg[:140],
            "level": level,
            "model": model,
            "ms": int(elapsed * 1000),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_saved": round(cost, 6),
            "answer": content[:300],
        })

    # Normalize messages: flatten content + convert tool_call formats
    norm_messages = normalize_messages_for_ollama(messages)

    # build the forward payload: pass through tools/options from the client
    forward = {"model": model, "messages": norm_messages}
    for k in ("tools", "tool_choice", "format", "keep_alive", "options"):
        if k in body:
            forward[k] = body[k]
    # map common OpenAI options into ollama options
    opts = dict(forward.get("options") or {})
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"),
                     ("max_tokens", "num_predict"), ("seed", "seed"),
                     ("stop", "stop"), ("frequency_penalty", "frequency_penalty"),
                     ("presence_penalty", "presence_penalty")):
        if src in body and dst not in opts:
            opts[dst] = body[src]
    if opts:
        forward["options"] = opts

    if stream:
        collected = []

        async def stream_response():
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA}/api/chat",
                        json={**forward, "stream": True},
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                msg = chunk.get("message", {}) or {}
                                token = msg.get("content", "") or ""
                                tool_calls_raw = msg.get("tool_calls")
                                tool_calls = tc_ollama_to_openai(tool_calls_raw) if tool_calls_raw else None
                                done = chunk.get("done", False)
                                collected.append(token)
                                delta = {"content": token}
                                if tool_calls:
                                    delta["tool_calls"] = tool_calls
                                    delta["content"] = None
                                sse = {
                                    "id": "chatcmpl-router",
                                    "object": "chat.completion.chunk",
                                    "model": model,
                                    "choices": [
                                        {
                                            "delta": delta,
                                            "finish_reason": ("tool_calls" if tool_calls else "stop") if done else None,
                                            "index": 0,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(sse)}\n\n"
                                if done:
                                    elapsed = time.time() - t_start
                                    record("".join(collected), elapsed)
                                    yield "data: [DONE]\n\n"
                            except Exception as inner:
                                print(f"  [Router] stream parse error: {inner}")
                                continue
            except Exception as e:
                import traceback
                print(f"  [Router] stream error: {type(e).__name__}: {e}")
                traceback.print_exc()
                err = {"error": {"message": str(e), "type": type(e).__name__}}
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                r = await client.post(
                    f"{OLLAMA}/api/chat",
                    json={**forward, "stream": False},
                )
                data = r.json()
        except Exception as e:
            import traceback
            print(f"  [Router] forward error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return JSONResponse(
                {"error": {"message": str(e), "type": type(e).__name__}},
                status_code=502,
            )
        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        tool_calls_raw = msg.get("tool_calls")
        if tool_calls_raw:
            print(f"  [Router] RAW tool_calls from Ollama: {json.dumps(tool_calls_raw, ensure_ascii=False)[:400]}")
        tool_calls = tc_ollama_to_openai(tool_calls_raw) if tool_calls_raw else None
        elapsed = time.time() - t_start
        record(content, elapsed)
        assistant_msg = {"role": "assistant", "content": content if not tool_calls else None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            print(f"  [Router] tool_calls → OpenAI: {json.dumps(tool_calls, ensure_ascii=False)[:400]}")
        return JSONResponse(
                {
                    "id": "chatcmpl-router",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "message": assistant_msg,
                            "finish_reason": "tool_calls" if tool_calls else "stop",
                            "index": 0,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": est_tokens(content),
                        "total_tokens": tokens_in + est_tokens(content),
                    },
                }
            )


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Router · Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg:       #07080c;
    --bg2:     #0e1018;
    --panel:   #11141f;
    --panel2:  #161a26;
    --border:  #252a3a;
    --text:    #e8eaf2;
    --muted:   #707587;
    --dim:     #454a5e;
    --accent:  #a78bfa;
    --accent2: #818cf8;
    --green:   #34d399;
    --blue:    #60a5fa;
    --yellow:  #fbbf24;
    --red:     #f87171;
    --pink:    #f472b6;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    overflow-x: hidden;
    font-feature-settings: 'cv11','ss01','ss03';
    -webkit-font-smoothing: antialiased;
  }
  body::before {
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(circle at 15% 10%, rgba(167,139,250,.12) 0%, transparent 40%),
      radial-gradient(circle at 85% 30%, rgba(96,165,250,.10) 0%, transparent 45%),
      radial-gradient(circle at 50% 100%, rgba(244,114,182,.08) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  /* HEADER */
  header {
    position: sticky; top: 0;
    background: rgba(7,8,12,.82);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex; align-items: center; gap: 16px;
    z-index: 100;
  }
  .logo {
    display: flex; align-items: center; gap: 12px;
    font-weight: 800; font-size: 1.05rem;
    letter-spacing: -.02em;
  }
  .logo-mark {
    width: 30px; height: 30px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--pink));
    display: grid; place-items: center;
    font-size: .9rem;
    color: #0b0e1c;
    box-shadow: 0 0 20px rgba(167,139,250,.4);
  }
  .pill {
    padding: 4px 10px;
    border-radius: 999px;
    font-size: .72rem; font-weight: 600;
    display: inline-flex; align-items: center; gap: 6px;
  }
  .pill.live { background: rgba(52,211,153,.1); color: var(--green); border: 1px solid rgba(52,211,153,.25); }
  .pill.live .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1; transform:scale(1)} 50%{opacity:.4; transform:scale(.8)} }

  .spacer { flex: 1; }
  .uptime { font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: .82rem; }

  /* LAYOUT */
  .container { position: relative; z-index: 1; max-width: 1320px; margin: 0 auto; padding: 28px 24px 60px; }

  .section-title {
    font-size: .72rem; text-transform: uppercase; letter-spacing: .14em;
    color: var(--muted); font-weight: 600; margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
  }
  .section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
  }

  /* HERO CARDS */
  .hero-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
  }
  .hero-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    position: relative;
    overflow: hidden;
    transition: transform .2s, border-color .2s;
  }
  .hero-card:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.35); }
  .hero-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: .5;
  }
  .hero-card .label {
    display: flex; align-items: center; gap: 8px;
    font-size: .72rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted); margin-bottom: 8px;
  }
  .hero-card .icon { font-size: 1rem; }
  .hero-card .value {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -.03em;
    line-height: 1; margin-bottom: 4px;
  }
  .hero-card .sub { font-size: .8rem; color: var(--muted); }
  .hero-card.featured {
    background: linear-gradient(135deg, rgba(167,139,250,.12), rgba(244,114,182,.08));
    border-color: rgba(167,139,250,.3);
  }
  .hero-card.featured .value {
    background: linear-gradient(135deg, var(--accent), var(--pink));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  /* MAIN GRID */
  .main-grid {
    display: grid;
    grid-template-columns: 1.15fr .85fr;
    gap: 24px;
    margin-bottom: 32px;
  }
  @media (max-width: 960px) { .main-grid, .hero-grid { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 640px) { .main-grid, .hero-grid { grid-template-columns: 1fr; } }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
  }
  .panel-header {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;
  }
  .panel-header h2 { font-size: .95rem; font-weight: 600; letter-spacing: -.01em; }
  .panel-header .hint { font-size: .75rem; color: var(--dim); }

  /* CHAT BOX */
  .chat-input { display: flex; gap: 10px; }
  .chat-input input {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 14px 16px;
    border-radius: 12px;
    font-size: .95rem; font-family: inherit;
    outline: none;
    transition: border-color .2s, box-shadow .2s;
  }
  .chat-input input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(167,139,250,.15);
  }
  .btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #07080c; border: none;
    padding: 0 22px; border-radius: 12px;
    font-weight: 700; font-size: .9rem; font-family: inherit;
    cursor: pointer;
    transition: transform .15s, box-shadow .2s;
    box-shadow: 0 4px 20px rgba(167,139,250,.3);
  }
  .btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 25px rgba(167,139,250,.45); }
  .btn:disabled { opacity: .5; cursor: not-allowed; transform: none; box-shadow: none; }

  .chat-result { margin-top: 16px; display: none; }
  .chat-result.show { display: block; animation: slideIn .3s; }
  @keyframes slideIn { from {opacity:0; transform:translateY(6px)} to {opacity:1; transform:none} }
  .chat-meta {
    display: flex; flex-wrap: wrap; gap: 10px;
    margin-bottom: 10px;
    font-size: .78rem;
  }
  .meta-chip {
    background: var(--bg2);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 8px;
    color: var(--muted);
  }
  .meta-chip strong { color: var(--text); font-weight: 600; }
  .chat-answer {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    font-size: .9rem;
    line-height: 1.65;
    max-height: 260px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .chat-answer::-webkit-scrollbar { width: 6px; }
  .chat-answer::-webkit-scrollbar-track { background: transparent; }
  .chat-answer::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* CHART PANEL */
  .chart-container { position: relative; height: 240px; }
  .legend-list { display: grid; gap: 10px; margin-top: 18px; }
  .legend-item {
    display: flex; align-items: center; gap: 10px;
    font-size: .84rem;
    padding: 10px 12px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    transition: transform .15s;
  }
  .legend-item:hover { transform: translateX(2px); }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend-label { flex: 1; }
  .legend-label strong { display: block; font-weight: 600; font-size: .85rem; text-transform: capitalize; }
  .legend-label .model-name { color: var(--muted); font-size: .72rem; font-family: 'JetBrains Mono', monospace; }
  .legend-count { color: var(--muted); font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: .85rem; }

  /* BADGES */
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: .7rem; font-weight: 600;
    text-transform: capitalize;
  }
  .badge::before { content: ''; width: 5px; height: 5px; border-radius: 50%; }
  .badge.rapido       { background: rgba(52,211,153,.12); color: var(--green); }
  .badge.rapido::before       { background: var(--green); }
  .badge.medio        { background: rgba(96,165,250,.12); color: var(--blue); }
  .badge.medio::before        { background: var(--blue); }
  .badge.razonamiento { background: rgba(251,191,36,.12); color: var(--yellow); }
  .badge.razonamiento::before { background: var(--yellow); }
  .badge.complejo     { background: rgba(248,113,113,.12); color: var(--red); }
  .badge.complejo::before     { background: var(--red); }

  /* HISTORY TABLE */
  .history-table-wrap {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
  }
  .history-header {
    padding: 18px 22px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .history-header h2 { font-size: .95rem; font-weight: 600; }
  .history-header .btn-refresh {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: .78rem; font-family: inherit;
    cursor: pointer;
    display: flex; align-items: center; gap: 6px;
    transition: all .15s;
  }
  .history-header .btn-refresh:hover { border-color: var(--accent); color: var(--accent); }
  .table-scroll { max-height: 520px; overflow-y: auto; }
  .table-scroll::-webkit-scrollbar { width: 8px; }
  .table-scroll::-webkit-scrollbar-track { background: transparent; }
  .table-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; }
  thead th {
    position: sticky; top: 0;
    background: var(--panel);
    font-size: .7rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--dim);
    padding: 12px 20px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  tbody td {
    padding: 14px 20px;
    font-size: .85rem;
    border-bottom: 1px solid rgba(37,42,58,.5);
    vertical-align: middle;
  }
  tbody tr { transition: background .15s; }
  tbody tr:hover td { background: rgba(167,139,250,.04); }
  .col-time { color: var(--dim); font-family: 'JetBrains Mono', monospace; font-size: .78rem; white-space: nowrap; }
  .col-query { max-width: 360px; color: var(--text); }
  .col-query .truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .col-model { font-family: 'JetBrains Mono', monospace; font-size: .76rem; color: var(--muted); white-space: nowrap; }
  .col-num { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--muted); text-align: right; white-space: nowrap; }
  .col-cost { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--green); text-align: right; white-space: nowrap; font-weight: 600; }
  .empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--dim);
  }
  .empty-icon { font-size: 2.5rem; margin-bottom: 10px; opacity: .4; }

  /* Models reference panel */
  .models-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-top: 18px;
  }
  .model-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    transition: transform .15s, border-color .15s;
  }
  .model-card:hover { transform: translateY(-2px); border-color: var(--accent); }
  .model-card .mc-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .model-card .mc-name { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--text); font-weight: 500; }
  .model-card .mc-equiv { font-size: .7rem; color: var(--muted); margin-bottom: 8px; }
  .model-card .mc-equiv strong { color: var(--pink); }
  .model-card .mc-cost { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--dim); }

  footer {
    text-align: center;
    color: var(--dim);
    font-size: .75rem;
    margin-top: 30px;
    padding: 20px;
  }
  footer a { color: var(--accent); text-decoration: none; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">⚡</div>
    <span>Smart Router</span>
  </div>
  <span class="pill live"><span class="dot"></span>Live</span>
  <div class="spacer"></div>
  <span class="uptime" id="uptime">uptime 0s</span>
</header>

<div class="container">

  <!-- Hero cards -->
  <div class="section-title">📊 Resumen</div>
  <div class="hero-grid">
    <div class="hero-card">
      <div class="label"><span class="icon">📨</span>Total Requests</div>
      <div class="value" id="h-total">0</div>
      <div class="sub" id="h-avg">0 ms avg</div>
    </div>
    <div class="hero-card featured">
      <div class="label"><span class="icon">💰</span>Ahorro Estimado</div>
      <div class="value" id="h-saved">$0.00</div>
      <div class="sub">vs cloud equivalente</div>
    </div>
    <div class="hero-card">
      <div class="label"><span class="icon">🔤</span>Tokens Procesados</div>
      <div class="value" id="h-tokens">0</div>
      <div class="sub"><span id="h-tokens-in">0</span> in · <span id="h-tokens-out">0</span> out</div>
    </div>
    <div class="hero-card">
      <div class="label"><span class="icon">🎯</span>Modelo Top</div>
      <div class="value" id="h-top" style="font-size: 1.3rem; padding-top:10px">—</div>
      <div class="sub" id="h-top-sub">sin datos</div>
    </div>
  </div>

  <!-- Main grid: chat + chart -->
  <div class="main-grid">
    <div class="panel">
      <div class="panel-header">
        <h2>🧪 Probar query</h2>
        <span class="hint">enter para enviar</span>
      </div>
      <div class="chat-input">
        <input type="text" id="q-input" placeholder="Escribí una pregunta..." onkeydown="if(event.key==='Enter') sendQuery()" autofocus />
        <button class="btn" id="q-btn" onclick="sendQuery()">Enviar →</button>
      </div>
      <div class="chat-result" id="chat-result">
        <div class="chat-meta" id="chat-meta"></div>
        <div class="chat-answer" id="chat-answer"></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2>📈 Distribución por nivel</h2>
      </div>
      <div class="chart-container">
        <canvas id="pie-chart"></canvas>
      </div>
      <div class="legend-list" id="legend-list"></div>
    </div>
  </div>

  <!-- History -->
  <div class="section-title">📜 Historial de requests</div>
  <div class="history-table-wrap">
    <div class="history-header">
      <h2>Últimas 200 requests</h2>
      <button class="btn-refresh" onclick="loadData()">↻ Actualizar</button>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Hora</th>
            <th>Query</th>
            <th>Nivel</th>
            <th>Modelo</th>
            <th style="text-align:right">Tokens</th>
            <th style="text-align:right">Tiempo</th>
            <th style="text-align:right">Ahorro</th>
          </tr>
        </thead>
        <tbody id="history-body">
          <tr><td colspan="7" class="empty"><div class="empty-icon">📭</div>Sin requests todavía</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Models reference -->
  <div style="margin-top: 32px">
    <div class="section-title">🧠 Modelos disponibles</div>
    <div class="panel">
      <div class="models-grid" id="models-grid"></div>
    </div>
  </div>

  <footer>
    Smart Router · ruteo inteligente local con Ollama ·
    <a href="/docs">API docs</a>
  </footer>

</div>

<script>
const COLORS = {
  rapido: '#34d399',
  medio: '#60a5fa',
  razonamiento: '#fbbf24',
  complejo: '#f87171',
};
const ICONS = { rapido:'⚡', medio:'🔵', razonamiento:'🟡', complejo:'🔴' };

let chart;
let lastData = null;

function fmtUptime(s) {
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}
function fmtCost(c) {
  if (c === 0) return '$0.00';
  if (c < 0.001) return '$' + c.toFixed(6);
  if (c < 0.01) return '$' + c.toFixed(5);
  if (c < 1) return '$' + c.toFixed(4);
  return '$' + c.toFixed(2);
}
function fmtNum(n) {
  if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n/1000).toFixed(1) + 'K';
  return String(n);
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderChart(stats) {
  const labels = Object.keys(stats);
  const counts = labels.map(k => stats[k].count);
  const hasData = counts.some(c => c > 0);
  const ctx = document.getElementById('pie-chart').getContext('2d');
  const data = {
    labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
    datasets: [{
      data: hasData ? counts : [1,1,1,1],
      backgroundColor: labels.map(l => COLORS[l] || '#888'),
      borderColor: '#11141f',
      borderWidth: 3,
      hoverOffset: 8,
    }]
  };
  if (chart) {
    chart.data = data;
    chart.update();
  } else {
    chart = new Chart(ctx, {
      type: 'doughnut',
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '62%',
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0e1018',
            titleColor: '#e8eaf2',
            bodyColor: '#a1a7bd',
            borderColor: '#252a3a',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
            callbacks: hasData ? {} : { label: () => 'Sin datos' }
          }
        }
      }
    });
  }
}

async function loadData() {
  try {
    const r = await fetch('/api/history');
    const d = await r.json();
    lastData = d;

    document.getElementById('uptime').textContent = 'uptime ' + fmtUptime(d.totals.uptime);

    // Hero cards
    document.getElementById('h-total').textContent = d.totals.count;
    document.getElementById('h-saved').textContent = fmtCost(d.totals.cost_saved);
    document.getElementById('h-tokens').textContent = fmtNum(d.totals.tokens_in + d.totals.tokens_out);
    document.getElementById('h-tokens-in').textContent = fmtNum(d.totals.tokens_in);
    document.getElementById('h-tokens-out').textContent = fmtNum(d.totals.tokens_out);

    // avg ms overall
    const totalMs = Object.values(d.stats).reduce((a,s) => a + s.ms_total, 0);
    const avg = d.totals.count ? Math.round(totalMs / d.totals.count) : 0;
    document.getElementById('h-avg').textContent = avg + ' ms avg';

    // top model
    let topKey = null, topCount = 0;
    for (const [k,s] of Object.entries(d.stats)) {
      if (s.count > topCount) { topCount = s.count; topKey = k; }
    }
    document.getElementById('h-top').textContent = topKey ? (ICONS[topKey] + ' ' + topKey) : '—';
    document.getElementById('h-top-sub').textContent = topKey ? (d.models[topKey] + ' · ' + topCount + ' uses') : 'sin datos';

    // chart + legend
    renderChart(d.stats);
    const legend = document.getElementById('legend-list');
    legend.innerHTML = Object.entries(d.stats).map(([k,s]) => `
      <div class="legend-item">
        <div class="legend-dot" style="background:${COLORS[k]}"></div>
        <div class="legend-label">
          <strong>${k}</strong>
          <span class="model-name">${d.models[k]}</span>
        </div>
        <div class="legend-count">${s.count}</div>
      </div>
    `).join('');

    // models grid
    const mg = document.getElementById('models-grid');
    mg.innerHTML = Object.entries(d.models).map(([k,m]) => {
      const eq = d.cloud_equiv[k] || {};
      const avgLvl = d.avg_ms[k] || 0;
      const st = d.stats[k];
      return `
        <div class="model-card">
          <div class="mc-head">
            <span class="badge ${k}">${ICONS[k]} ${k}</span>
            <span style="font-size:.7rem;color:var(--dim);font-family:'JetBrains Mono',monospace">${avgLvl}ms</span>
          </div>
          <div class="mc-name">${m}</div>
          <div class="mc-equiv">equivale a: <strong>${eq.name || '—'}</strong></div>
          <div class="mc-cost">$${eq.in?.toFixed(2) || '0'}/M in · $${eq.out?.toFixed(2) || '0'}/M out</div>
          <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-size:.75rem">
            <span style="color:var(--muted)">${st.count} calls</span>
            <span style="color:var(--green);font-weight:600;font-family:'JetBrains Mono',monospace">${fmtCost(st.cost_saved)}</span>
          </div>
        </div>
      `;
    }).join('');

    // history
    const tbody = document.getElementById('history-body');
    if (!d.history.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty"><div class="empty-icon">📭</div>Sin requests todavía</td></tr>';
    } else {
      tbody.innerHTML = d.history.map(h => `
        <tr>
          <td class="col-time">${h.time}</td>
          <td class="col-query"><div class="truncate" title="${escHtml(h.query)}">${escHtml(h.query)}</div></td>
          <td><span class="badge ${h.level}">${ICONS[h.level]||''} ${h.level}</span></td>
          <td class="col-model">${h.model}</td>
          <td class="col-num">${h.tokens_in}↓ ${h.tokens_out}↑</td>
          <td class="col-num">${h.ms}ms</td>
          <td class="col-cost">${fmtCost(h.cost_saved)}</td>
        </tr>
      `).join('');
    }
  } catch(e) { console.error(e); }
}

async function sendQuery() {
  const input = document.getElementById('q-input');
  const q = input.value.trim();
  if (!q) return;
  const btn = document.getElementById('q-btn');
  const resultDiv = document.getElementById('chat-result');
  const metaDiv = document.getElementById('chat-meta');
  const answerDiv = document.getElementById('chat-answer');
  btn.disabled = true;
  btn.textContent = 'Pensando...';
  resultDiv.classList.add('show');
  metaDiv.innerHTML = '<span class="meta-chip">Clasificando...</span>';
  answerDiv.textContent = '';
  try {
    const t0 = Date.now();
    const r = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'smart-router', messages: [{role:'user', content: q}], stream: false})
    });
    const d = await r.json();
    const elapsed = Date.now() - t0;
    const model = d.model;
    const content = d.choices?.[0]?.message?.content || '(sin respuesta)';
    const usage = d.usage || {};
    metaDiv.innerHTML = `
      <span class="meta-chip">modelo <strong>${model}</strong></span>
      <span class="meta-chip">⏱ <strong>${elapsed}ms</strong></span>
      <span class="meta-chip">🔤 <strong>${usage.total_tokens||0}</strong> tok</span>
    `;
    answerDiv.textContent = content;
    input.value = '';
    setTimeout(loadData, 300);
  } catch(e) {
    metaDiv.innerHTML = '<span class="meta-chip" style="color:var(--red)">Error: ' + escHtml(e.message) + '</span>';
  }
  btn.disabled = false;
  btn.textContent = 'Enviar →';
}

loadData();
setInterval(loadData, 4000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6061)
