#!/usr/bin/env python3
"""server.py - OpenAI-compatible server supporting IDE file attachments."""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from agent import run_agent

SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}
_FALLBACK_CLIENT: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lifecycle of shared async HTTP connection pool."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        app.state.http_client = client
        yield


app = FastAPI(title="Jimmy OpenAI Compatible Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_METADATA = {
    "id": "llama3.1-8B",
    "object": "model",
    "created": 1788822782,
    "name": "Llama 3.1 8B (ChatJimmy)",
    "owned_by": "AMD",
}


def get_client(request: Request) -> httpx.AsyncClient:
    """Provides client pool, falling back to singleton if uninitialized."""
    client: Optional[httpx.AsyncClient] = getattr(
        request.app.state, "http_client", None
    )
    if client is not None:
        return client

    global _FALLBACK_CLIENT
    if _FALLBACK_CLIENT is None:
        _FALLBACK_CLIENT = httpx.AsyncClient(timeout=180.0)
    return _FALLBACK_CLIENT


def normalize_content(content: Any) -> str:
    """Normalizes string or multipart list payloads into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Standard OpenAI text part
                if "text" in item and isinstance(item["text"], str):
                    name = item.get("name") or item.get("filename")
                    if name:
                        parts.append(
                            f"--- File: {name} ---\n{item['text']}\n--- End ---"
                        )
                    else:
                        parts.append(item["text"])
                # OpenCode / IDE file reference block
                elif "file" in item:
                    f_info = item["file"]
                    if isinstance(f_info, dict):
                        fname = f_info.get("name", "attached_file")
                        fcontent = f_info.get("content", "")
                        parts.append(
                            f"--- File: {fname} ---\n{fcontent}\n--- End ---"
                        )
                    else:
                        parts.append(str(f_info))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(content)


# --- Request Schemas with Permissive Config ---


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Optional[str] = "user"
    content: Optional[Any] = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = "llama3.1-8B"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    enable_tools: Optional[bool] = True


# --- Endpoints ---


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [MODEL_METADATA]}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return MODEL_METADATA


@app.post("/v1/chat/reset")
async def reset_session(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Clear conversational memory."""
    session_id = x_session_id or "default"
    SESSION_STORE.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    client: httpx.AsyncClient = Depends(get_client),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    # Normalize complex message formats into plain text role/content dicts
    incoming: List[Dict[str, str]] = [
        {
            "role": msg.role or "user",
            "content": normalize_content(msg.content),
        }
        for msg in body.messages
    ]

    session_id = x_session_id or "default"

    # If the IDE sends the entire history on each turn, adopt it directly
    if len(incoming) > 1:
        convo_history = incoming
        SESSION_STORE[session_id] = list(incoming)
    else:
        store = SESSION_STORE.setdefault(session_id, [])
        store.extend(incoming)
        convo_history = list(store)

    output = await run_agent(
        messages=convo_history,
        model=body.model or "llama3.1-8B",
        client=client,
        enable_tools=bool(body.enable_tools),
    )

    # Cache assistant turn in session
    SESSION_STORE[session_id].append({"role": "assistant", "content": output})

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if body.stream:
        async def event_generator():
            delta_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": body.model or "llama3.1-8B",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": output},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(delta_chunk)}\n\n"

            stop_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": body.model or "llama3.1-8B",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )

    prompt_toks = sum(len(m["content"].split()) for m in convo_history)
    comp_toks = len(output.split())

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": body.model or "llama3.1-8B",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 4100))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
