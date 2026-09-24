# ChatJimmy OpenAI Compatible Model Wrapper

<img width="256" height="256" alt="Jimbo" src="https://github.com/user-attachments/assets/39a9266a-981a-4d31-bcff-16f46a99ae39" />

> It's lil wonky sometimes, but pur dang fast. 


## 🚀 Getting Started

```bash
# [⚠️ OPTIONAL] Use a virtual environment with `python3 -m venv venv` then `source venv/bin/activate`

# Otherwise, install the dependencies in requirements.txt file with:
python3 -m pip install -r requirements.txt

# Start the server
python3 server.py
```

**NOTE:** The default port is 4100. You can override it by setting the `PORT` environment variable. Also, if you wish to iterate on it, since it uses uvicorn, it will automatically reload on code changes to it.



## 📋 TODO

- [x] Fix memory issue. Validate with `tests.sh` script.
- [x] Add some general NLP recognitiion to set up some `@mcp.tool()` commands.
- [x] Implement tool call: `get_current_weather`. and `get_current_time` (really just for debugging the memory issue).
- [x] Implement tool call: `read_file` and `write_file` for file I/O.
- [x] Implement tool call: `list_directory` for directory listing.
- [x] Implement tool call: `execute_shell` for executing shell commands.
- [x] Implement tool call: `search_code` for introspection.
- [ ] Ignore MCP agent context linger. *PARTIALLY COMPLETE*. To be safe, I recommend disabling the MCP agent context in the config file when using this model wrapper for now.
- [ ] Implement tool call: `get_file_info` for file metadata *MAY NOT BE NECESSARY, SINCE WE CAN USE `execute_shell` with `grep`, `stat`, `file`, etc.



## Made up some model details
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.1-8B"
      "object": "model",
      "created":1788822782
      "name": "Llama 3.1 8B (ChatJimmy)",
      "description": "LLaMA 3.1 8B model running on Taalas HC1 with ChatJimmy",
      "owned_by": "AMD"
    }
  ]
},

```
