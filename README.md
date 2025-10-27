# MCP Aggregator Server

Unified MCP interface that proxies requests to multiple backend MCP servers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client                              │
│              (Claude, IDE, etc.)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Connect to single endpoint
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Aggregator MCP Server (Port 8003)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Unified MCP Interface                               │   │
│  │  - 19 tools total (2 health + 10 memory + 7 vector) │   │
│  │  - Handles routing internally                        │   │
│  │  - Single /mcp/sse & /mcp/messages endpoint          │   │
│  └──────────────────────────────────────────────────────┘   │
└────────┬──────────────────────────────────────────────────┬──┘
         │                                                  │
         │ HTTP Proxy                                       │ HTTP Proxy
         ▼                                                  ▼
┌──────────────────────┐                        ┌──────────────────────┐
│  ZepAI Memory Server │                        │  LTM Vector Server   │
│  (Port 8002)         │                        │  (Port 8000)         │
│                      │                        │                      │
│ - Knowledge Graph    │                        │ - Vector Database    │
│ - Conversation Memory│                        │ - Code Indexing      │
│ - 10 tools           │                        │ - 7 tools            │
└──────────────────────┘                        └──────────────────────┘
```

## Features

- **Unified Interface**: Single MCP endpoint for all connected servers
- **Transparent Proxying**: Automatically routes requests to appropriate backend servers
- **Health Monitoring**: Built-in health checks for all connected servers
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Error Handling**: Comprehensive error handling and logging
- **Extensible**: Easy to add new backend servers

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment** (edit `.env`):
```env
# Aggregator Server
AGGREGATOR_HOST=0.0.0.0
AGGREGATOR_PORT=8003

# Memory Server (FastMCP Server)
MEMORY_SERVER_URL=http://localhost:8002
MEMORY_SERVER_TIMEOUT=30

# Graph Server (for future use)
GRAPH_SERVER_URL=http://localhost:8000
GRAPH_SERVER_TIMEOUT=30
```

## Running

### Start all servers in order:

**Terminal 1 - LTM Vector Server (Port 8000)**:
```bash
cd LTM
python mcp_server/server_streamable_http.py
```

**Terminal 2 - ZepAI FastMCP Server (Port 8002)**:
```bash
cd ZepAI/fastmcp_server
python server_http.py
```
*Note: This automatically loads the Memory Layer and exposes both FastAPI + MCP on port 8002*

**Terminal 3 - MCP Aggregator (Port 8003)**:
```bash
cd mcp-aggregator
python aggregator_server.py
```

**See [START_SERVERS.md](START_SERVERS.md) for detailed startup guide.**

## Available Tools

### Health & Status
- `health_check()` - Check health of all connected servers
- `get_server_info()` - Get information about connected servers

### Memory Server Tools (Port 8002)

#### Search
- `memory_search(query, project_id, limit, use_llm_classification)` - Search knowledge graph
- `memory_search_code(query, project_id, limit)` - Search code memories

#### Ingest
- `memory_ingest_text(text, project_id, metadata)` - Ingest plain text
- `memory_ingest_code(code, language, project_id, metadata)` - Ingest code
- `memory_ingest_json(data, project_id, metadata)` - Ingest JSON data
- `memory_ingest_conversation(conversation, project_id)` - Ingest conversation

#### Admin
- `memory_get_stats(project_id)` - Get project statistics
- `memory_get_cache_stats()` - Get cache statistics

### LTM Vector Server Tools (Port 8000)

#### Repository Processing
- `ltm_process_repo(repo_path)` - Process repository for vector indexing

#### Vector Search
- `ltm_query_vector(query, top_k)` - Query vector database for semantic code search
- `ltm_search_file(filepath)` - Search for specific file in vector database

#### File Management
- `ltm_add_file(filepath)` - Add file to vector database
- `ltm_delete_by_filepath(filepath)` - Delete file from vector database
- `ltm_delete_by_uuids(uuids)` - Delete vectors by UUIDs

#### Code Analysis
- `ltm_chunk_file(file_path)` - Chunk file using AST-based chunking

## Testing

### 1. Check Server Health
```bash
curl http://localhost:8003/mcp/sse
```

### 2. Access OpenAPI Docs
```
http://localhost:8003/docs
```

### 3. Test a Tool via MCP
```bash
# Using MCP client
mcp-client http://localhost:8003/mcp health_check
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGGREGATOR_HOST` | `0.0.0.0` | Aggregator server host |
| `AGGREGATOR_PORT` | `8003` | Aggregator server port |
| `MEMORY_SERVER_URL` | `http://localhost:8002` | Memory server URL |
| `MEMORY_SERVER_TIMEOUT` | `30` | Memory server timeout (seconds) |
| `GRAPH_SERVER_URL` | `http://localhost:8000` | Graph server URL |
| `GRAPH_SERVER_TIMEOUT` | `30` | Graph server timeout (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_RETRIES` | `3` | Max retries for failed requests |
| `RETRY_DELAY` | `1` | Delay between retries (seconds) |
| `HEALTH_CHECK_INTERVAL` | `30` | Health check interval (seconds) |

## Adding New Backend Servers

To add a new backend server (e.g., Graph Server):

1. **Update `config.py`**:
```python
GRAPH_SERVER_URL = os.getenv("GRAPH_SERVER_URL", "http://localhost:8000")
GRAPH_SERVER_TIMEOUT = int(os.getenv("GRAPH_SERVER_TIMEOUT", "30"))
```

2. **Update `mcp_client.py`**:
```python
class AggregatorClients:
    def __init__(self):
        # ... existing clients ...
        self.graph_client = MCPServerClient(
            "Graph Server",
            config.GRAPH_SERVER_URL,
            config.GRAPH_SERVER_TIMEOUT
        )
```

3. **Add tools in `aggregator_server.py`**:
```python
@mcp.tool()
async def graph_query(cypher: str) -> Dict[str, Any]:
    """Query Neo4j graph database"""
    clients = await get_clients()
    return await clients.graph_client.proxy_request(
        "POST",
        "/query",
        json_data={"cypher": cypher},
        retries=config.MAX_RETRIES
    )
```

## Troubleshooting

### Connection Refused
- Ensure all backend servers are running
- Check URLs in `.env` file
- Verify ports are not blocked by firewall

### Timeout Errors
- Increase `MEMORY_SERVER_TIMEOUT` or `GRAPH_SERVER_TIMEOUT` in `.env`
- Check backend server performance
- Verify network connectivity

### Health Check Failing
- Run `health_check()` tool to diagnose
- Check backend server logs
- Verify backend servers are responding

## Development

### Project Structure
```
mcp_aggregator/
├── aggregator_server.py    # Main MCP server
├── config.py               # Configuration management
├── mcp_client.py           # HTTP clients for backend servers
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── __init__.py             # Package initialization
└── README.md               # This file
```

### Adding Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
logger.error("Error")
```

## Future Enhancements

- [ ] Add Graph/Vector DB server integration
- [ ] Implement caching layer
- [ ] Add request rate limiting
- [ ] Implement server load balancing
- [ ] Add metrics/monitoring
- [ ] Support for server discovery
- [ ] WebSocket support for real-time updates

## License

Same as parent project (Innocody)
