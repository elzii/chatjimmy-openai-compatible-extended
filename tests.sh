#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:4100}"
SESSION="test-suite-$(date +%s)"

send_prompt() {
  local prompt="$1"
  curl -s -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: ${SESSION}" \
    -d "$(jq -n --arg p "$prompt" '{"messages": [{"role": "user", "content": $p}]}')"
}

echo "=== 1. Resetting Test Session (${SESSION}) ==="
curl -s -X POST "${BASE_URL}/v1/chat/reset" -H "X-Session-ID: ${SESSION}" | jq .
echo

echo "=== 2. Testing Memory Turn 1 ==="
echo "User: My favorite language is Rust."
RES1=$(send_prompt "My favorite language is Rust.")
echo -n "Assistant: "
echo "$RES1" | jq -r '.choices[0].message.content // .detail // .'
echo

echo "=== 3. Testing Memory Recall Turn 2 ==="
echo "User: What is my favorite language?"
RES2=$(send_prompt "What is my favorite language?")
echo -n "Assistant: "
echo "$RES2" | jq -r '.choices[0].message.content // .detail // .'
echo

echo "=== 4. Testing Streaming (SSE Emulation as used by OpenCode) ==="
echo "User: Count from 1 to 3."
curl -N -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: ${SESSION}" \
  -d '{"messages": [{"role": "user", "content": "Count from 1 to 3."}], "stream": true}' | grep -E '^data: '
echo
