## 🧩 Modellake as an MCP Tool: Full Integration Guide

### 1. Install MCP SDK

```bash
pip install mcp
```

---

### 2. Create MCP Server Wrapper (`modellake_mcp_server.py`)

```python
import asyncio
from mcp import types as mcp_types
from mcp.server.lowlevel import Server
import modellake.cli_api as mlake

app = Server("modellake-mcp-server")

TOOLS = {
    "download": lambda args: mlake.download(args["resource"], args.get("mode","scratch"), args.get("dest","./data/")),
    "extract_table": lambda args: mlake.extract_table(args["resource"], args.get("mode","scratch"), args.get("dest","./data/")),
    "quality_control": lambda args: mlake.quality_control(args["mode"], args.get("dest","./data/")),
    "extract_relatedness": lambda args: mlake.extract_relatedness(args.get("resource","paper")),
    "table_search": lambda args: mlake.table_search(args["input_table"], args.get("method","dense"), args.get("directory","./data/")),
    "plot_analysis": lambda args: mlake.plot_analysis(),
    "repeat_experiments": lambda args: mlake.repeat_experiments(args["method"], args["resource"], args["relatedness"]),
}

@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(name=name, description=f"Modellake API: {name}", args_schema=mcp_types.JsonSchema(type="object"))
        for name in TOOLS
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.Content]:
    if name in TOOLS:
        result = TOOLS[name](arguments)
        return [mcp_types.TextContent(type="text", text=result)]
    return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]
```

* **`list_tools()`** exposes available tools (“download”, “extract\_table”, etc.) ([Medium][2])
* **`call_tool()`** executes the requested Werkzeug and returns output ([philschmid.de][3])

---

### 3. Launch the MCP Server

For local integration, use **stdio transport**:

```bash
python modellake_mcp_server.py
```

This server now speaks MCP over JSON-RPC via stdin/stdout ([philschmid.de][3]).

---

### 4. Register MCP Server with Agent (Client Side)

Example using OpenAI Agents SDK:

```python
from mcp.server.lowlevel import MCPServerStdio
from openai_agents import Agent  # hypothetical SDK

server = MCPServerStdio(command="python", args=["modellake_mcp_server.py"])
agent = Agent(name="ModellakeAgent", instructions="...", mcp_servers=[server])
```

On startup, the agent:

1. Calls `list_tools()` → discovers "download", "extract\_table", etc.
2. When planning, may call `call_tool("download", {...})`
3. The server executes and returns Modellake results ([Google GitHub][4], [Cloud Native Deep Dive][5])

---

### 5. How it Works: End-to-End Flow

```
Agent (MCP Client)
      │── list_tools() ──► Discover Modellake functions
      │
      │── call_tool("download", {resource, mode, dest}) ──►
                     └── MCP Server ──► modellake.download()
                             │
                         returns stdout log
                             │
      ◄────────── tool result ────────────
```

* Agent can inspect results, chain calls, build workflows ([Medium][6], [philschmid.de][3])

---

### 6. Why This Matters

Our tools can be embedded and used by all MCP supported clients, which will help to broadcast our developed table discovery methods!
