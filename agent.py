"""agent.py - FastMCP tool runner with gated tools and conversational prompt."""

from datetime import datetime
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import zoneinfo

import httpx
from mcp.server.fastmcp import FastMCP

API_URL = "https://chatjimmy.ai/api/chat"
DEFAULT_MODEL = "llama3.1-8B"

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.7",
    "content-type": "application/json",
    "origin": "https://chatjimmy.ai",
    "referer": "https://chatjimmy.ai/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
}

mcp = FastMCP("JimmyTools")


# -----------------------------------------------------------------------------
# FastMCP Tools (with docstring-level gating)
# -----------------------------------------------------------------------------


@mcp.tool()
def get_current_time(timezone: str = "America/Los_Angeles") -> str:
    """
    Get current date and time.
    Use ONLY when the user explicitly asks for the current time or date.
    """
    try:
        tz_clean = timezone.strip().replace(" ", "_")
        city_map = {
            "San_Diego": "America/Los_Angeles",
            "Seattle": "America/Los_Angeles",
            "San_Francisco": "America/Los_Angeles",
            "New_York": "America/New_York",
            "London": "Europe/London",
            "Paris": "Europe/Paris",
            "Tokyo": "Asia/Tokyo",
        }
        resolved = city_map.get(tz_clean, tz_clean)
        now = datetime.now(zoneinfo.ZoneInfo(resolved))
        return now.strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception as exc:
        return f"Error resolving timezone '{timezone}': {exc}"


@mcp.tool()
def get_current_weather(city: str) -> str:
    """
    Get current weather conditions for a city.
    Use ONLY when the user explicitly asks about the weather.
    """
    return json.dumps(
        {"city": city, "temperature": "70°F", "condition": "Clear"}
    )


@mcp.tool()
def evaluate_math(expression: str) -> str:
    """
    Safely calculate a basic arithmetic expression.
    Use ONLY when the user asks for a math calculation.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(char in allowed for char in expression):
        return "Error: Unsupported characters in mathematical expression."
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def execute_shell(command: str, timeout_seconds: int = 30) -> str:
    """
    Execute a local shell command.
    Use ONLY when the user explicitly asks to run a command or script.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd(),
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        out = []
        if stdout:
            out.append(f"STDOUT:\n{stdout}")
        if stderr:
            out.append(f"STDERR:\n{stderr}")
        out.append(f"Exit Code: {proc.returncode}")
        return "\n\n".join(out)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_seconds} seconds."
    except Exception as exc:
        return f"Command execution failed: {exc}"


# -----------------------------------------------------------------------------
# Upstream Communication & Helpers
# -----------------------------------------------------------------------------


async def get_tool_schemas() -> List[Dict[str, Any]]:
    """Inspect and format FastMCP tool signatures."""
    tools = await mcp.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema,
        }
        for t in tools
    ]


def parse_jimmy_body(body: str) -> Tuple[str, Dict[str, Any]]:
    """Strip telemetry stats block from the ChatJimmy raw text response."""
    content, stats = body, {}
    if "<|stats|>" in body:
        content, _, rest = body.partition("<|stats|>")
        try:
            stats = json.loads(rest.split("<|/stats|>")[0])
        except json.JSONDecodeError:
            stats = {}
    return content.rstrip("\n"), stats


async def call_upstream(
    messages: List[Dict[str, str]],
    system_prompt: str,
    client: httpx.AsyncClient,
    model: str = DEFAULT_MODEL,
) -> Tuple[int, str]:
    """Execute asynchronous POST to chatjimmy.ai."""
    payload = {
        "messages": messages,
        "chatOptions": {
            "selectedModel": model,
            "systemPrompt": system_prompt,
            "topK": 8,
        },
        "attachment": None,
    }
    try:
        response = await client.post(API_URL, headers=HEADERS, json=payload)
        return response.status_code, response.text
    except Exception as exc:
        return 0, f"Upstream error: {exc}"


def find_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls supporting code blocks or multiple JSON objects."""
    tool_calls: List[Dict[str, Any]] = []
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidates = code_blocks if code_blocks else [text]

    decoder = json.JSONDecoder()
    for candidate in candidates:
        idx = 0
        cand_len = len(candidate)
        while idx < cand_len:
            match = re.search(r"[\{\[]", candidate[idx:])
            if not match:
                break
            idx += match.start()
            try:
                obj, end_idx = decoder.raw_decode(candidate[idx:])
                idx += end_idx
                if isinstance(obj, dict) and "tool" in obj:
                    tool_calls.append(obj)
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict) and "tool" in item:
                            tool_calls.append(item)
            except json.JSONDecodeError:
                idx += 1

    return tool_calls


# -----------------------------------------------------------------------------
# Agent Loop
# -----------------------------------------------------------------------------


async def run_agent(
    messages: List[Dict[str, str]],
    model: str,
    client: httpx.AsyncClient,
    enable_tools: bool = True,
    max_turns: int = 5,
) -> str:
    """Agent orchestrator: loops between LLaMA 3.1 and FastMCP tools."""
    history: List[Dict[str, str]] = [dict(m) for m in messages]
    system_prompt = ""

    if enable_tools:
        schemas = await get_tool_schemas()
        system_prompt = (
            "You are a friendly, direct conversational AI assistant.\n"
            "When the user greets you, introduces themselves, chats casually, "
            "or asks about previous messages, reply directly in natural text.\n\n"
            "You have access to tools for specific external tasks:\n"
            f"{json.dumps(schemas, indent=2)}\n\n"
            "TOOL CALLING RULES:\n"
            "- Call a tool ONLY when the user's latest request specifically "
            "requires that tool's capability.\n"
            "- When you need to call a tool, respond with ONLY a JSON block:\n"
            "```json\n"
            '{"tool": "tool_name", "arguments": {"arg": "val"}}\n'
            "```\n"
            "- Otherwise, respond with plain text."
        )

    for _ in range(max_turns):
        status, raw_body = await call_upstream(
            messages=history,
            system_prompt=system_prompt,
            client=client,
            model=model,
        )
        if status != 200:
            return f"[Upstream HTTP error {status}: {raw_body}]"

        content, _ = parse_jimmy_body(raw_body)
        tool_calls = find_tool_calls(content) if enable_tools else []

        if not tool_calls:
            return content

        history.append({"role": "assistant", "content": content})

        results: List[str] = []
        for tc in tool_calls:
            t_name = str(tc.get("tool", ""))
            t_args = tc.get("arguments", {})
            print(
                f"[FastMCP] Invoking tool: {t_name}({t_args})",
                file=sys.stderr,
            )
            try:
                res = await mcp.call_tool(t_name, t_args)
                if isinstance(res, list):
                    res_str = "\n".join(
                        getattr(item, "text", str(item)) for item in res
                    )
                else:
                    res_str = str(res)
            except Exception as exc:
                res_str = f"Error: {exc}"
            results.append(f"• Tool `{t_name}` result: {res_str}")

        history.append(
            {
                "role": "user",
                "content": (
                    "[Tool Observation Data]:\n"
                    + "\n".join(results)
                    + "\n\nAnswer ONLY the user's most recent request directly "
                    "using this data. Do not re-answer previous questions."
                ),
            }
        )

    return "Agent error: Maximum tool iterations exceeded."
