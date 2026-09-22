#!/usr/bin/env python3
"""server.py - OpenAI-compatible FastAPI server with automatic memory."""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import run_agent

# Persistent session store: session_id -> list of messages
SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}


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
    """Dependency provider guaranteeing an initialized httpx.AsyncClient."""
    client: Optional[httpx.AsyncClient] = getattr(
        request.app.state, "http_client", None
    )
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="HTTP client pool is not initialized.",
        )
    return client


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "llama3.1-8B"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    enable_tools: Optional[bool] = True


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [MODEL_METADATA]}


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    client: httpx.AsyncClient = Depends(get_client),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    incoming: List[Dict[str, str]] = [
        {"role": msg.role, "content": msg.content} for msg in body.messages
    ]

    # Session identifier defaults to "default" so simple curl calls retain memory
    session_id = x_session_id or "default"

    # If client manages full history (sends multiple messages), adopt it.
    # If client sends single messages incrementally (e.g. curl), append to session.
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

    # Save assistant response to session store
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
