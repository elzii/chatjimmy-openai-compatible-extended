"""agent.py - FastMCP tool runner with gated filesystem and coding tools."""

from datetime import datetime
import json
import os
from pathlib import Path
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
# FastMCP Tools (Docstring-Gated)
# -----------------------------------------------------------------------------


@mcp.tool()
def read_file(path: str, max_lines: int = 500) -> str:
    """
    Read contents of a local text or code file.
    Use when the user asks to read, explain, inspect, or summarize a file.
    """
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return f"Error: File not found: {path}"
    if not target.is_file():
        return f"Error: Path is not a file: {path}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        clipped = lines[:max_lines]
        result = "\n".join(clipped)
        if total > max_lines:
            result += f"\n\n[... Truncated {total - max_lines} lines ...]"
        return result
    except Exception as exc:
        return f"Error reading file {path}: {exc}"


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """
    List files and directories in a given folder.
    Use when the user asks what files exist or to explore project layout.
    """
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return f"Error: Path does not exist: {path}"
    if not target.is_dir():
        return f"Error: Path is not a directory: {path}"

    entries: List[str] = []
    try:
        for item in sorted(target.iterdir()):
            if item.name.startswith(".git"):
                continue
            marker = "/" if item.is_dir() else ""
            size = (
                f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            )
            entries.append(f"{item.name}{marker}{size}")
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as exc:
        return f"Error listing directory: {exc}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Create a new file or completely overwrite an existing file.
    Use when the user asks to create, save, or write a file.
    """
    target = Path(path).expanduser().resolve()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}."
    except Exception as exc:
        return f"Error writing file {path}: {exc}"


@mcp.tool()
def edit_file(path: str, target: str, replacement: str) -> str:
    """
    Replace an exact code snippet in a file with new replacement text.
    Use when the user asks to update, modify, or patch code in an existing file.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return f"Error: File does not exist: {path}"
    try:
        original = p.read_text(encoding="utf-8")
        if target not in original:
            return (
                f"Error: Target text not found in {path}. "
                "Ensure exact match including whitespace and indentation."
            )
        count = original.count(target)
        updated = original.replace(target, replacement, 1)
        p.write_text(updated, encoding="utf-8")
        return f"Successfully updated {path} (1 of {count} matches)."
    except Exception as exc:
        return f"Error editing file {path}: {exc}"


@mcp.tool()
def search_code(
    pattern: str,
    path: str = ".",
    max_results: int = 30,
) -> str:
    """
    Search for text or regex pattern across code files.
    Use when searching for functions, variables, or keywords in a project.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Error: Path does not exist: {path}"

    rg_installed = subprocess.run(
        "command -v rg", shell=True, capture_output=True
    ).returncode == 0

    if rg_installed:
        cmd = [
            "rg",
            "--max-count", "5",
            "--max-columns", "150",
            "--glob", "!.git",
            "--glob", "!node_modules",
            "--glob", "!__pycache__",
            "--glob", "!.venv",
            pattern,
            str(root),
        ]
    else:
        cmd = [
            "grep",
            "-rnI",
            "--exclude-dir={.git,node_modules,__pycache__,.venv}",
            f"--max-count={max_results}",
            pattern,
            str(root),
        ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = [
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        ]
        if not lines:
            return f"No matches found for: '{pattern}'"
        clipped = lines[:max_results]
        out = "\n".join(clipped)
        if len(lines) > max_results:
            out += f"\n\n[... Truncated at {max_results} matches ...]"
        return out
    except Exception as exc:
        return f"Search execution failed: {exc}"


@mcp.tool()
def execute_shell(command: str, timeout_seconds: int = 30) -> str:
    """
    Execute a local shell command.
    Use ONLY when the user explicitly asks to run a command, script, or test.
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
            "You are a friendly, direct conversational software engineering "
            "assistant.\n"
            "When the user greets you, introduces themselves, chats casually, "
            "or asks about previous conversation history, reply directly in "
            "natural text.\n\n"
            "You have direct access to tools for local files, shell commands, "
            "and system data:\n"
            f"{json.dumps(schemas, indent=2)}\n\n"
            "TOOL CALLING RULES:\n"
            "- When asked to inspect, read, or explain a file (e.g. 'what does "
            "server.py do?'), use `read_file`.\n"
            "- When asked what files exist or to look around, use "
            "`list_directory`.\n"
            "- When asked to run a command or script, use `execute_shell`.\n"
            "- When you need to call a tool, respond with ONLY a JSON block:\n"
            "```json\n"
            '{"tool": "tool_name", "arguments": {"arg": "val"}}\n'
            "```\n"
            "- Otherwise, respond directly in plain text."
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
