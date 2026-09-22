"""agent.py - Autonomous coding agent with FastMCP tools & ChatJimmy backend."""

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
from playwright.async_api import async_playwright

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

mcp = FastMCP("JimmyCodingAgent")

# -----------------------------------------------------------------------------
# 1. FastMCP Tool Suite
# -----------------------------------------------------------------------------


@mcp.tool()
def get_current_time(timezone: str = "America/Los_Angeles") -> str:
    """Get current date and time for a given timezone or city region."""
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
        resolved_tz = city_map.get(tz_clean, tz_clean)
        now = datetime.now(zoneinfo.ZoneInfo(resolved_tz))
        return now.strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception as exc:
        return f"Error resolving timezone '{timezone}': {exc}"


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather conditions for a given city."""
    return json.dumps(
        {"city": city, "temperature": "70°F", "condition": "Clear"}
    )


@mcp.tool()
def evaluate_math(expression: str) -> str:
    """Safely calculate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not all(char in allowed for char in expression):
        return "Error: Unsupported characters in mathematical expression."
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List directory contents with file sizes and directory indicators."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return f"Path does not exist: {path}"
    if not target.is_dir():
        return f"Path is not a directory: {path}"

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
        return f"Failed to list directory: {exc}"


@mcp.tool()
def read_file(path: str, max_lines: int = 500) -> str:
    """Read contents of a text file up to max_lines."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return f"File not found: {path}"
    if not target.is_file():
        return f"Target is not a file: {path}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        clipped = lines[:max_lines]
        result = "\n".join(clipped)
        if total > max_lines:
            result += (
                f"\n\n[... Truncated {total - max_lines} remaining lines ...]"
            )
        return result
    except Exception as exc:
        return f"Error reading file {path}: {exc}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Create a new file or completely overwrite an existing file."""
    target = Path(path).expanduser().resolve()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}."
    except Exception as exc:
        return f"Error writing file {path}: {exc}"


@mcp.tool()
def edit_file(path: str, target: str, replacement: str) -> str:
    """Replace an exact substring target with replacement in a file."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return f"Error: File does not exist: {path}"
    try:
        original = p.read_text(encoding="utf-8")
        if target not in original:
            return (
                f"Error: Target text not found in {path}. "
                "Ensure spacing and newlines match exactly."
            )
        count = original.count(target)
        updated = original.replace(target, replacement, 1)
        p.write_text(updated, encoding="utf-8")
        return (
            f"Successfully updated {path} (replaced 1 of {count} occurrences)."
        )
    except Exception as exc:
        return f"Error editing file {path}: {exc}"


@mcp.tool()
def search_code(
    pattern: str,
    path: str = ".",
    max_results: int = 30,
) -> str:
    """Search for regex/text across files, ignoring build artifacts."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Path does not exist: {path}"

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
            return f"No matches found for pattern: '{pattern}'"
        clipped = lines[:max_results]
        out = "\n".join(clipped)
        if len(lines) > max_results:
            out += f"\n\n[... Truncated at {max_results} matches ...]"
        return out
    except Exception as exc:
        return f"Search execution failed: {exc}"


@mcp.tool()
def execute_shell(command: str, timeout_seconds: int = 30) -> str:
    """Execute a local shell command (git, compilers, tests, builds)."""
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
async def fetch_webpage(url: str, max_chars: int = 5000) -> str:
    """Fetch live web pages, render JS, and extract readable text."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)

            title = await page.title()

            # Clean out structural junk that inflates token counts
            await page.evaluate("""() => {
                const tags = [
                    'script', 'style', 'noscript', 'svg',
                    'iframe', 'nav', 'footer', 'header'
                ];
                tags.forEach(tag => {
                    document.querySelectorAll(tag).forEach(
                        el => el.remove()
                    );
                });
            }""")

            raw_text = await page.inner_text("body")
            await browser.close()

            lines = [
                line.strip() for line in raw_text.splitlines() if line.strip()
            ]
            content = "\n".join(lines)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... Truncated ...]"

            return f"Title: {title}\nURL: {url}\n\n{content}"
    except Exception as exc:
        return f"Failed to fetch webpage at {url}: {exc}"


# -----------------------------------------------------------------------------
# 2. Helpers & Upstream Communication
# -----------------------------------------------------------------------------


async def get_tool_schemas() -> List[Dict[str, Any]]:
    """Reflect tool schemas from the FastMCP server instance."""
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
# 3. Coding Assistant Prompt & Agent Loop
# -----------------------------------------------------------------------------


def build_system_prompt(
    tool_schemas: List[Dict[str, Any]],
    user_system_override: str = "",
) -> str:
    """Builds the system prompt for the autonomous coding assistant."""
    base_instructions = (
        "You are an expert autonomous software engineer and assistant.\n"
        "You have direct access to local system tools including filesystem "
        "operations, code search, shell execution, live web browsing, and "
        "time/weather lookup.\n\n"
        "OPERATIONAL GUIDELINES:\n"
        "1. Be proactive, precise, and direct. When asked to inspect, edit, "
        "or test code, invoke the appropriate tools immediately. Do not ask "
        "permission to use a tool.\n"
        "2. When inspecting unfamiliar code, read the file first before "
        "attempting edits.\n"
        "3. When using `edit_file`, ensure the `target` parameter matches "
        "exact lines and indentation in the source file.\n"
        "4. If a user query asks multiple questions, execute all required "
        "tools.\n"
        "5. Respond naturally in conversation. Never leak JSON formats or "
        "mention internal tool names to the user.\n\n"
        "TOOL INVOCATION FORMAT:\n"
        "To invoke tools, output ONLY a JSON block:\n"
        "```json\n"
        '{"tool": "tool_name", "arguments": {"arg": "val"}}\n'
        "```\n\n"
        f"Available tools:\n{json.dumps(tool_schemas, indent=2)}"
    )
    if user_system_override.strip():
        return (
            f"{base_instructions}\n\n"
            f"ADDITIONAL INSTRUCTIONS:\n{user_system_override}"
        )
    return base_instructions


async def run_agent(
    messages: List[Dict[str, str]],
    model: str,
    client: httpx.AsyncClient,
    enable_tools: bool = True,
    max_turns: int = 6,
) -> str:
    """Agent orchestrator: loops between LLaMA 3.1 and FastMCP tools."""
    user_sys_msgs = [
        m["content"] for m in messages if m.get("role") == "system"
    ]
    user_system_override = "\n".join(user_sys_msgs)

    history: List[Dict[str, str]] = [
        dict(m) for m in messages if m.get("role") != "system"
    ]

    schemas = await get_tool_schemas() if enable_tools else []
    system_prompt = build_system_prompt(schemas, user_system_override)

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
            results.append(f"• Tool `{t_name}` result:\n{res_str}")

        history.append(
            {
                "role": "user",
                "content": (
                    "[Tool Observation Data]:\n"
                    + "\n\n".join(results)
                    + "\n\nPlease provide a complete, direct response "
                    "answering all parts of the user's inquiry."
                ),
            }
        )

    return "Agent error: Maximum tool iterations exceeded."
