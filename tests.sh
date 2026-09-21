#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${1:-http://localhost:4100}"

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required but not installed." >&2
  exit 1
fi

send_prompt() {
  local prompt="$1"
  curl -s -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$prompt" '{"messages": [{"role": "user", "content": $p}]}')"
}

echo "=== 1. Resetting Session Memory ==="
curl -s -X POST "${BASE_URL}/v1/chat/reset" | jq .
echo

echo "=== 2. Teaching Context (Memory Turn 1) ==="
echo "User: My favorite fruit is mango."
RES1=$(send_prompt "My favorite fruit is mango.")
echo -n "Assistant: "
echo "$RES1" | jq -r '.choices[0].message.content // .detail // .'
echo

echo "=== 3. Testing Context Recall (Memory Turn 2) ==="
echo "User: What is my favorite fruit?"
RES2=$(send_prompt "What is my favorite fruit?")
echo -n "Assistant: "
echo "$RES2" | jq -r '.choices[0].message.content // .detail // .'
echo

echo "=== 4. Testing Multi-Tool Execution (FastMCP) ==="
echo "User: What is the weather in Seattle and what is 512 / 8?"
RES3=$(send_prompt "What is the weather in Seattle and what is 512 / 8?")
echo -n "Assistant: "
echo "$RES3" | jq -r '.choices[0].message.content // .detail // .'
echo
