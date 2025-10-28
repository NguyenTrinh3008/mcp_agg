# MCP Aggregator - Complete Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [API Reference](#api-reference)
5. [Agent Tools (MCP)](#agent-tools-mcp)
6. [Innocody Trigger Endpoints](#innocody-trigger-endpoints)
7. [Admin Endpoints](#admin-endpoints)
8. [Configuration](#configuration)
9. [Usage Examples](#usage-examples)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**MCP Aggregator** là một unified MCP (Model Context Protocol) server tập trung hóa quyền truy cập vào nhiều backend services:

- **ZepAI Memory Server** (Port 8002): Knowledge Graph + Conversation Memory
- **LTM Vector Server** (Port 8000): Vector Database + Code Indexing + Knowledge Graph

### Key Features

✅ **Unified Interface**: Single MCP endpoint cho tất cả services  
✅ **Agent-Friendly**: 4 search/query tools cho AI agents  
✅ **Innocody Integration**: HTTP trigger endpoints cho Innocody Engine  
✅ **Admin Tools**: Health checks, stats, server info  
✅ **Retry Logic**: Exponential backoff cho failed requests  
✅ **CORS Support**: Cross-origin requests enabled  

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Agent / Client                        │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MCP Aggregator (Port 8003)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4 Agent Tools (MCP):                                │   │
│  │  - memory_search()                                   │   │
│  │  - ltm_query_vector()                                │   │
│  │  - ltm_search_file()                                 │   │
│  │  - ltm_find_code()                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  HTTP Trigger Endpoints (Innocody):                  │   │
│  │  - /triggers/memory/ingest                           │   │
│  │  - /triggers/ltm/*                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Admin Endpoints:                                    │   │
│  │  - /admin/health                                     │   │
│  │  - /admin/info                                       │   │
│  │  - /admin/memory/stats                               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────┬─────────────────────────────┬──────────────────┘
             │                             │
             ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│  ZepAI Memory Server    │   │   LTM Vector Server         │
│  (Port 8002)            │   │   (Port 8000)               │
│  - Knowledge Graph      │   │   - Vector Database         │
│  - Conversation Memory  │   │   - Code Indexing           │
│                         │   │   - Knowledge Graph         │
└─────────────────────────┘   └─────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- ZepAI Memory Server running on port 8002
- LTM Vector Server running on port 8000

### Installation

```bash
cd c:\Users\Lenovo\Desktop\Innocody\mcp-aggregator

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env with your configuration
notepad .env
```

### Environment Variables

Create `.env` file:

```env
# Aggregator Server
AGGREGATOR_HOST=0.0.0.0
AGGREGATOR_PORT=8003

# Memory Server (ZepAI)
MEMORY_SERVER_URL=http://localhost:8002
MEMORY_SERVER_TIMEOUT=30

# LTM Server (Vector Database)
LTM_SERVER_URL=http://localhost:8000
LTM_SERVER_TIMEOUT=30

# Retry settings
MAX_RETRIES=3
RETRY_DELAY=1

# Logging
LOG_LEVEL=INFO
```

### Start Server

```bash
# Start aggregator
python aggregator_server.py

# Or use uvicorn directly
uvicorn aggregator_server:app --host 0.0.0.0 --port 8003
```

Server will be available at:
- **MCP Endpoint**: `http://localhost:8003`
- **Swagger UI**: `http://localhost:8003/docs`
- **Health Check**: `http://localhost:8003/admin/health`

---

## API Reference

### Base URL

```
http://localhost:8003
```

### Response Format

All endpoints return JSON:

```json
{
  "status": "success",
  "data": { ... },
  "error": null
}
```

Error response:

```json
{
  "status": "error",
  "error": "Error message",
  "detail": "Detailed error information"
}
```

---

## Agent Tools (MCP)

These tools are exposed via MCP protocol for AI agents to use.

### 1. memory_search

Search conversation memory and knowledge graph.

**Parameters:**
- `query` (string, required): Search query
- `limit` (int, optional): Max results, default 10
- `project_id` (string, optional): Filter by project
- `use_llm_classification` (bool, optional): Use LLM for classification, default false

**Example:**

```python
# Via MCP client
result = await mcp_client.call_tool(
    "memory_search",
    query="authentication logic",
    limit=5
)
```

**Response:**

```json
{
  "results": [
    {
      "content": "User authentication is handled by JWT tokens...",
      "score": 0.95,
      "metadata": {
        "type": "conversation",
        "timestamp": "2025-10-28T10:00:00Z"
      }
    }
  ],
  "total": 5
}
```

---

### 2. ltm_query_vector

Search code semantically using vector embeddings.

**Parameters:**
- `query` (string, required): Natural language query
- `top_k` (int, optional): Number of results, default 10

**Example:**

```python
result = await mcp_client.call_tool(
    "ltm_query_vector",
    query="function that validates user input",
    top_k=5
)
```

**Response:**

```json
{
  "results": [
    {
      "code": "def validate_input(data):\n    ...",
      "file_path": "/path/to/file.py",
      "score": 0.89,
      "metadata": {
        "language": "python",
        "function_name": "validate_input"
      }
    }
  ]
}
```

---

### 3. ltm_search_file

Get all indexed chunks of a specific file.

**Parameters:**
- `filepath` (string, required): Absolute path to file

**Example:**

```python
result = await mcp_client.call_tool(
    "ltm_search_file",
    filepath="C:\\Users\\Lenovo\\Desktop\\haha\\snake_game.py"
)
```

**Response:**

```json
{
  "file_path": "C:\\Users\\Lenovo\\Desktop\\haha\\snake_game.py",
  "chunks": [
    {
      "uuid": "abc-123",
      "code": "class Snake:\n    def __init__(self)...",
      "start_line": 10,
      "end_line": 50
    }
  ],
  "total_chunks": 5
}
```

---

### 4. ltm_find_code

Find code entities (functions, classes) in knowledge graph.

**Parameters:**
- `query` (string, required): Search query for code entities

**Example:**

```python
result = await mcp_client.call_tool(
    "ltm_find_code",
    query="Snake class"
)
```

**Response:**

```json
{
  "functions": [
    {
      "name": "move",
      "file_path": "/path/to/snake_game.py",
      "start_line": 25,
      "end_line": 35,
      "complexity": 3
    }
  ],
  "classes": [
    {
      "name": "Snake",
      "file_path": "/path/to/snake_game.py",
      "methods": ["__init__", "move", "grow"]
    }
  ]
}
```

---

## Innocody Trigger Endpoints

These HTTP endpoints are called by Innocody Engine to update memory and code index. **Not exposed as MCP tools.**

### Memory Triggers

#### POST /triggers/memory/ingest

Ingest content into memory.

**Request Body:**

```json
{
  "content": "User wants to implement authentication",
  "content_type": "text",
  "project_id": "my-project",
  "metadata": {
    "source": "user_request",
    "timestamp": "2025-10-28T10:00:00Z"
  }
}
```

**Content Types:**
- `text`: Plain text
- `code`: Code snippets
- `json`: Structured data
- `conversation`: Conversation history

**Example:**

```bash
curl -X POST "http://localhost:8003/triggers/memory/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Implement JWT authentication",
    "content_type": "text",
    "project_id": "my-app"
  }'
```

**Response:**

```json
{
  "status": "success",
  "message": "Content ingested successfully",
  "memory_id": "mem-123"
}
```

---

### LTM Triggers

#### POST /triggers/ltm/process_repo

Index entire repository into graph and vector DB.

**Query Parameters:**
- `repo_path` (string, required): Absolute path to repository

**Example:**

```bash
curl -X POST "http://localhost:8003/triggers/ltm/process_repo?repo_path=C:\Users\Lenovo\Desktop\haha"
```

**Response:**

```json
{
  "status": "success",
  "message": "Successfully indexed repository",
  "processed_files": 15,
  "path": "C:\\Users\\Lenovo\\Desktop\\haha"
}
```

---

#### POST /triggers/ltm/add_file

Add single file to vector database.

**Query Parameters:**
- `filepath` (string, required): Absolute path to file

**Example:**

```bash
curl -X POST "http://localhost:8003/triggers/ltm/add_file?filepath=C:\path\to\file.py"
```

---

#### PUT /triggers/ltm/update_files

Update multiple files in graph and vector DB.

**Request Body:**

```json
{
  "file_updates": [
    {
      "filepath": "C:\\path\\to\\file1.py",
      "action": "update"
    },
    {
      "filepath": "C:\\path\\to\\file2.py",
      "action": "delete"
    }
  ]
}
```

---

#### DELETE /triggers/ltm/delete

Delete from vector database by filepath or UUIDs.

**Query Parameters:**
- `filepath` (string, optional): Delete all chunks of file
- `uuids` (array, optional): Delete specific chunks by UUID

**Example:**

```bash
# Delete by filepath
curl -X DELETE "http://localhost:8003/triggers/ltm/delete?filepath=C:\path\to\file.py"

# Delete by UUIDs
curl -X DELETE "http://localhost:8003/triggers/ltm/delete" \
  -H "Content-Type: application/json" \
  -d '["uuid1", "uuid2", "uuid3"]'
```

---

#### POST /triggers/ltm/chunk_file

Chunk file using AST parser.

**Query Parameters:**
- `file_path` (string, required): Absolute path to file

**Example:**

```bash
curl -X POST "http://localhost:8003/triggers/ltm/chunk_file?file_path=C:\path\to\file.py"
```

**Response:**

```json
{
  "file_path": "C:\\path\\to\\file.py",
  "chunks": [
    {
      "type": "function",
      "name": "validate_input",
      "start_line": 10,
      "end_line": 25,
      "code": "def validate_input(data):\n    ..."
    }
  ],
  "total_chunks": 5
}
```

---

## Admin Endpoints

### GET /admin/health

Check health status of all connected servers.

**Example:**

```bash
curl http://localhost:8003/admin/health
```

**Response:**

```json
{
  "status": "healthy",
  "aggregator": "healthy",
  "servers": {
    "memory_server": true,
    "ltm_server": true
  }
}
```

---

### GET /admin/info

Get information about all connected servers.

**Example:**

```bash
curl http://localhost:8003/admin/info
```

**Response:**

```json
{
  "aggregator": {
    "name": "Unified Knowledge Server",
    "version": "1.0.0",
    "url": "http://0.0.0.0:8003"
  },
  "connected_servers": {
    "zepai_memory_server": {
      "url": "http://localhost:8002",
      "status": "connected",
      "tools": ["memory_search"]
    },
    "ltm_server": {
      "url": "http://localhost:8000",
      "status": "connected",
      "tools": ["ltm_query_vector", "ltm_search_file", "ltm_find_code"]
    }
  }
}
```

---

### GET /admin/memory/stats

Get memory statistics.

**Query Parameters:**
- `stats_type` (string, required): "project" or "cache"
- `project_id` (string, optional): Required if stats_type=project

**Example:**

```bash
# Project stats
curl "http://localhost:8003/admin/memory/stats?stats_type=project&project_id=my-app"

# Cache stats
curl "http://localhost:8003/admin/memory/stats?stats_type=cache"
```

**Response:**

```json
{
  "project_id": "my-app",
  "total_memories": 150,
  "conversations": 25,
  "code_snippets": 75,
  "documents": 50
}
```

---

## Configuration

### Config Class

Located in `config.py`:

```python
class Config:
    # Aggregator Server
    AGGREGATOR_HOST = "0.0.0.0"
    AGGREGATOR_PORT = 8003
    AGGREGATOR_NAME = "Unified Knowledge Server"
    AGGREGATOR_VERSION = "1.0.0"
    
    # Memory Server (ZepAI)
    MEMORY_SERVER_URL = "http://localhost:8002"
    MEMORY_SERVER_TIMEOUT = 30
    
    # LTM Server (Vector Database)
    LTM_SERVER_URL = "http://localhost:8000"
    LTM_SERVER_TIMEOUT = 30
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # Health check
    HEALTH_CHECK_INTERVAL = 30  # seconds
```

### Retry Logic

All requests use exponential backoff:

```python
retries = 3
delay = 1  # second

# Retry delays: 1s, 2s, 4s
```

---

## Usage Examples

### Python Client

```python
import asyncio
from mcp import Client

async def main():
    # Connect to aggregator
    client = Client("http://localhost:8003")
    
    # Search memory
    result = await client.call_tool(
        "memory_search",
        query="authentication implementation",
        limit=5
    )
    print(result)
    
    # Search code
    code_result = await client.call_tool(
        "ltm_query_vector",
        query="JWT token validation",
        top_k=3
    )
    print(code_result)

asyncio.run(main())
```

### Innocody Engine Integration

```rust
// Rust code in Innocody Engine
async fn index_repository(repo_path: &str) -> Result<()> {
    let client = reqwest::Client::new();
    
    let response = client
        .post("http://localhost:8003/triggers/ltm/process_repo")
        .query(&[("repo_path", repo_path)])
        .send()
        .await?;
    
    let result: serde_json::Value = response.json().await?;
    println!("Indexed {} files", result["processed_files"]);
    
    Ok(())
}
```

### cURL Examples

```bash
# Health check
curl http://localhost:8003/admin/health

# Index repository
curl -X POST "http://localhost:8003/triggers/ltm/process_repo?repo_path=C:\my\repo"

# Ingest memory
curl -X POST "http://localhost:8003/triggers/memory/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User wants authentication",
    "content_type": "text",
    "project_id": "my-app"
  }'

# Get server info
curl http://localhost:8003/admin/info
```

---

## Troubleshooting

### Common Issues

#### 1. Connection Refused

**Error:**
```
Connection refused to http://localhost:8002
```

**Solution:**
- Ensure ZepAI Memory Server is running on port 8002
- Check `MEMORY_SERVER_URL` in `.env`

#### 2. Timeout Errors

**Error:**
```
Request timeout after 30 seconds
```

**Solution:**
- Increase timeout in `.env`:
  ```env
  MEMORY_SERVER_TIMEOUT=60
  LTM_SERVER_TIMEOUT=60
  ```

#### 3. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'fastmcp'
```

**Solution:**
```bash
pip install -r requirements.txt
```

#### 4. Port Already in Use

**Error:**
```
Address already in use: 8003
```

**Solution:**
```bash
# Change port in .env
AGGREGATOR_PORT=8004

# Or kill existing process
netstat -ano | findstr :8003
taskkill /PID <PID> /F
```

### Debug Mode

Enable detailed logging:

```env
LOG_LEVEL=DEBUG
```

View logs:

```bash
# Windows
type logs\aggregator.log

# Linux/Mac
tail -f logs/aggregator.log
```

### Health Check

```bash
# Check all services
curl http://localhost:8003/admin/health

# Expected response
{
  "status": "healthy",
  "aggregator": "healthy",
  "servers": {
    "memory_server": true,
    "ltm_server": true
  }
}
```

---

## Support

For issues or questions:
- Check logs in `logs/aggregator.log`
- Review Swagger UI at `http://localhost:8003/docs`
- Test individual endpoints with cURL
- Verify backend servers are running

---

**Version:** 1.0.0  
**Last Updated:** October 28, 2025  
**Author:** Innocody Team
