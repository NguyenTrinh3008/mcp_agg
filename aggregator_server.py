"""
MCP Aggregator Server
=====================
Unified MCP interface that proxies requests to multiple backend MCP servers.

Architecture:
- Aggregator Server (Port 8003) - This server
- ZepAI Memory Server (Port 8002) - Knowledge Graph + Conversation Memory
- LTM Vector Server (Port 8000) - Vector Database + Code Indexing + Knowledge Graph

Current Status:
    ZepAI Memory Server (8002) - READY (1 search tool)
    LTM Vector Server (8000) - READY (3 search tools)
    Total: 4 tools available for agents (search/query only)
    
    Innocody triggers (HTTP endpoints):
    - Memory ingest, LTM indexing, file operations available via HTTP
    - Admin tools (health, info, stats) available via HTTP

The Aggregator exposes all tools from both servers as a single MCP interface.
Clients connect to the Aggregator and can access all tools transparently.
"""

from fastmcp import FastMCP
import sys
import os
import uvicorn
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, status, Body
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import config and clients
from config import config
from mcp_client import get_clients, close_clients

# ============================================================================
# CREATE MCP SERVER
# ============================================================================

mcp = FastMCP(name=config.AGGREGATOR_NAME)

logger.info(f"Created MCP server: {mcp.name}")
logger.info("Status: ZepAI (8002) ✅ | LTM (8000) ✅ | Total: 4 search tools for agents")

# ==========================================================================
# INTERNAL HELPERS
# ==========================================================================


def _format_conversation_message(message: Any, max_chars: int = 240) -> str:
    if isinstance(message, dict):
        role = message.get("role", "unknown")
        content_text = message.get("content", "")
        if not isinstance(content_text, str):
            content_text = str(content_text)
    else:
        role = "message"
        content_text = str(message)

    content_text = content_text.strip()
    if len(content_text) > max_chars:
        content_text = content_text[: max_chars - 3] + "..."

    return f"{role}: {content_text}" if role else content_text


def _summarize_conversation_messages(
    messages: Any,
    max_messages: int = 6,
    max_chars_per_message: int = 240,
) -> tuple[List[str], bool, int]:
    if not isinstance(messages, list) or max_messages <= 0:
        return [], False, 0

    total = len(messages)
    if total <= max_messages:
        formatted = [
            _format_conversation_message(msg, max_chars_per_message)
            for msg in messages
        ]
        return formatted, False, total

    head = max(1, max_messages // 2)
    tail = max_messages - head
    head_messages = messages[:head]
    tail_messages = messages[-tail:]

    formatted = [
        _format_conversation_message(msg, max_chars_per_message)
        for msg in head_messages
    ]
    formatted.append(f"... ({total - max_messages} messages omitted) ...")
    formatted.extend(
        _format_conversation_message(msg, max_chars_per_message)
        for msg in tail_messages
    )

    return formatted, True, total


# ==========================================================================
# INTERNAL ADMIN FUNCTIONS (Not exposed as MCP tools)
# ==========================================================================

async def _health_check() -> Dict[str, Any]:
    """
    Internal function: Check health status of all connected servers
    Used by HTTP endpoints, not exposed to agents
    """
    clients = await get_clients()
    health_status = await clients.health_check_all()
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "servers": health_status,
        "aggregator": "healthy"
    }


async def _get_server_info() -> Dict[str, Any]:
    """
    Internal function: Get information about all connected servers
    Used by HTTP endpoints, not exposed to agents
    """
    clients = await get_clients()
    schemas = await clients.get_all_schemas()
    
    info = {
        "aggregator": {
            "name": config.AGGREGATOR_NAME,
            "version": config.AGGREGATOR_VERSION,
            "url": f"http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}",
        },
        "connected_servers": {}
    }
    
    # ZepAI Memory Server info
    if schemas.get("memory_server"):
        memory_schema = schemas["memory_server"]
        info["connected_servers"]["zepai_memory_server"] = {
            "url": config.MEMORY_SERVER_URL,
            "title": memory_schema.get("info", {}).get("title", "ZepAI Memory Layer"),
            "version": memory_schema.get("info", {}).get("version", "Unknown"),
            "endpoints_count": len(memory_schema.get("paths", {})),
            "tools": 2,
            "description": "Knowledge Graph + Conversation Memory"
        }
    
    # LTM Vector Server info
    if schemas.get("ltm_server"):
        ltm_schema = schemas["ltm_server"]
        info["connected_servers"]["ltm_vector_server"] = {
            "url": config.LTM_SERVER_URL,
            "title": ltm_schema.get("info", {}).get("title", "LTM API"),
            "version": ltm_schema.get("info", {}).get("version", "Unknown"),
            "endpoints_count": len(ltm_schema.get("paths", {})),
            "tools": 8,
            "description": "Vector Database + Code Indexing + Knowledge Graph"
        }
    
    return info


# ============================================================================
# AGENT TOOLS - SEARCH/QUERY ONLY (Exposed as MCP tools)
# ============================================================================

@mcp.tool()
async def memory_search(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 10,
    use_llm_classification: bool = False
) -> Dict[str, Any]:
    """
    Search in memory knowledge graph for relevant context
    
    Args:
        query: Search query string
        project_id: Project ID (optional)
        limit: Maximum number of results
        use_llm_classification: Use LLM for classification
    
    Returns:
        Search results from memory server
    """
    return await _memory_search(query, project_id, limit, use_llm_classification)


# ============================================================================
# INNOCODY TRIGGER FUNCTIONS - MEMORY (HTTP endpoints only)
# ============================================================================

async def _memory_ingest(
    content: str | Dict[str, Any],
    content_type: str = "text",
    project_id: Optional[str] = None,
    language: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Internal function: Ingest content into memory
    Called by Innocody engine triggers, not exposed to agents
    """
    clients = await get_clients()
    
    # Endpoint mapping
    endpoint_map = {
        "text": "/ingest/text",
        "code": "/ingest/code",
        "json": "/ingest/json",
        "conversation": "/ingest/message"  # Use simpler message endpoint
    }
    
    if content_type not in endpoint_map:
        return {
            "error": f"Invalid content_type: {content_type}",
            "valid_types": list(endpoint_map.keys())
        }
    
    # Build payload based on content type
    if content_type == "text":
        # Handle both string and dictionary input for text content
        if isinstance(content, dict):
            # If content is a dict, extract text and name
            text_content = content.get('text', str(content))
            name = content.get('name', 'text_content')
            
            # Handle conversation messages if present
            if 'messages' in content and isinstance(content['messages'], list):
                # Format messages as text
                messages = []
                for msg in content['messages']:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        msg_content = msg.get('content', '')
                        messages.append(f"{role}: {msg_content}")
                    else:
                        messages.append(str(msg))
                text_content = '\n'.join(messages)
                
                # Use session_id for name if available
                if 'session_id' in content:
                    name = f"Conversation {content['session_id']}"
        else:
            # If content is a string, use it as is
            text_content = str(content)
            name = 'text_content'
        
        # Build the final payload
        payload = {
            "name": metadata.get("name", name) if metadata else name,
            "text": text_content,
            "group_id": project_id,
            "source_description": metadata.get("source_description", "app") if metadata else "app"
        }
    elif content_type == "code":
        payload = {
            "code": content,
            "metadata": metadata or {}
        }
        if language:
            payload["language"] = language
        if project_id:
            payload["project_id"] = project_id
    elif content_type == "json":
        payload = {
            "data": content,
            "metadata": metadata or {}
        }
        if project_id:
            payload["project_id"] = project_id
    elif content_type == "conversation":
        # /ingest/message expects: {name: str, messages: List[str], group_id: Optional[str]}
        if isinstance(content, dict):
            session_id = content.get("session_id", "unknown_session")
            messages = content.get("messages", [])

            summary_messages, truncated, total_messages = _summarize_conversation_messages(
                messages
            )

            if not summary_messages:
                summary_messages = ["(no messages provided)"]

            name_suffix = " (summary)" if truncated else ""

            payload = {
                "name": f"Conversation {session_id}{name_suffix}",
                "messages": summary_messages,
                "group_id": project_id,
                "source_description": "innocody_conversation",
            }

            if truncated:
                payload["metadata"] = {
                    "total_messages": total_messages,
                    "summary_count": len(summary_messages),
                }
        else:
            return {"error": "conversation content_type requires dict with session_id and messages"}
    
    return await clients.memory_client.proxy_request(
        "POST",
        endpoint_map[content_type],
        json_data=payload,
        retries=config.MAX_RETRIES
    )


async def _memory_ingest_code_changes(
    diff_chunks: List[Dict[str, Any]],
    project_id: str,
    chat_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal function: Ingest DiffChunks into memory as code changes
    Called by Innocody engine triggers, not exposed to agents
    
    This properly converts DiffChunks to structured code change metadata
    and stores them in Neo4j knowledge graph with relationships.
    """
    clients = await get_clients()
    
    if not diff_chunks:
        return {"error": "No diff_chunks provided"}
    
    # Use the specialized diffchunks endpoint
    payload = {
        "project_id": project_id,
        "chat_id": chat_id or f"engine_{int(datetime.now().timestamp())}",
        "diff_chunks": diff_chunks
    }
    
    logger.info(f"Ingesting {len(diff_chunks)} DiffChunks as code changes to Neo4j")
    
    return await clients.memory_client.proxy_request(
        "POST",
        "/innocody/webhook/diffchunks",
        json_data=payload,
        retries=config.MAX_RETRIES
    )


async def _memory_stats(
    stats_type: str = "project",
    project_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal function: Get statistics from memory server
    Used by admin endpoints, not exposed to agents
    """
    clients = await get_clients()
    
    if stats_type == "project":
        if not project_id:
            return {"error": "project_id is required for project stats"}
        return await clients.memory_client.proxy_request(
            "GET",
            f"/stats/{project_id}",
            retries=config.MAX_RETRIES
        )
    elif stats_type == "cache":
        return await clients.memory_client.proxy_request(
            "GET",
            "/cache/stats",
            retries=config.MAX_RETRIES
        )
    else:
        return {
            "error": f"Invalid stats_type: {stats_type}",
            "valid_types": ["project", "cache"]
        }


# ============================================================================
# INTERNAL LTM SEARCH FUNCTIONS (Can be called from endpoint)
# ============================================================================

async def _ltm_query_vector(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Internal function: Semantic code search"""
    clients = await get_clients()
    return await clients.ltm_client.proxy_request(
        "GET",
        "/vectors/query",
        params={"query": query, "top_k": top_k},
        retries=config.MAX_RETRIES
    )


# Removed _ltm_search_file and _ltm_find_code
# All search functionality now consolidated in ltm_query_vector which uses /vectors/query
# This endpoint returns both vector results AND graph data in result_graph field


async def _memory_search(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 10,
    use_llm_classification: bool = False
) -> Dict[str, Any]:
    """Internal function: Memory search"""
    clients = await get_clients()
    payload = {
        "query": query,
        "limit": limit,
        "use_llm_classification": use_llm_classification
    }
    if project_id:
        payload["project_id"] = project_id
    return await clients.memory_client.proxy_request(
        "POST",
        "/search",
        json_data=payload,
        retries=config.MAX_RETRIES
    )


# ============================================================================
# AGENT TOOLS - LTM SEARCH/QUERY (Exposed as MCP tools)
# ============================================================================

@mcp.tool()
async def ltm_search(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search code semantically using unified vector + graph search.
    
    This is the ONLY LTM search tool you need. It returns:
    - results: Vector search results with code chunks, scores, metadata
    - result_graph: Knowledge graph data with entities, relationships, hierarchical context
    
    Args:
        query: Natural language query to find similar code (e.g., "where is pacman's maze?")
        top_k: Number of top results to return (default: 10)
    
    Returns:
        {
            "results": [
                {
                    "id": "uuid",
                    "score": 0.34,
                    "payload": {
                        "uuid": "uuid",
                        "filepath": "path/to/file.py",
                        "start_line": 18,
                        "end_line": 18,
                        "language": ".py",
                        "type": "function_definition",
                        "name": ["function_name"],
                        "code": "actual code content",
                        "hybrid_score": 0.99,
                        "rerank_score": 1.0
                    }
                }
            ],
            "result_graph": {
                "uuid": {
                    "labels": "Function|Class|Variable",
                    "file_path": "path/to/file.py",
                    "language": "python",
                    "name": "entity_name",
                    "ingoing_relations": {...},
                    "outgoing_relations": {...},
                    "hierarchical_context": [...]
                }
            }
        }
    
    Example:
        result = ltm_search("authentication logic", top_k=5)
        for item in result["results"]:
            print(f"File: {item['payload']['filepath']}")
            print(f"Code: {item['payload']['code']}")
            print(f"Score: {item['score']}")
        
        # Access graph data
        for uuid, entity in result["result_graph"].items():
            print(f"Entity: {entity['name']} ({entity['labels']})")
            print(f"Relations: {entity['outgoing_relations']}")
    """
    return await _ltm_query_vector(query, top_k)


# ============================================================================
# INNOCODY TRIGGER FUNCTIONS - LTM (HTTP endpoints only)
# ============================================================================

async def _ltm_process_repo(repo_path: str) -> Dict[str, Any]:
    """
    Internal function: Index repository into graph and vector DB
    Called by Innocody engine triggers, not exposed to agents
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/repos/process_repo/",
        params={"repo_path": repo_path},
        retries=config.MAX_RETRIES
    )


async def _ltm_add_file(filepath: str) -> Dict[str, Any]:
    """
    Internal function: Add file to vector database AND code graph
    Called by Innocody engine triggers, not exposed to agents
    
    Updates both:
    1. Vector DB - for semantic search
    2. Code Graph - for code relationships and structure
    """
    clients = await get_clients()
    
    # Step 1: Add to Vector DB
    vector_result = await clients.ltm_client.proxy_request(
        "POST",
        "/vectors/files",
        params={"filepath": filepath},
        retries=config.MAX_RETRIES
    )
    
    # Step 2: Add to Code Graph
    try:
        graph_result = await clients.ltm_client.proxy_request(
            "POST",
            "/graph/files/",
            json_data={
                "file_name": filepath,
                "file_action": "add",
                "file_name_rename": "",
                "line1": 0,
                "line2": 0,
                "lines_add": "",
                "lines_remove": "",
                "file_rename": "",
                "application_details": "Auto-add file to code graph"
            },
            retries=config.MAX_RETRIES
        )
        
        return {
            "status": "success",
            "message": f"File added to both vector DB and code graph: {filepath}",
            "vector_db": vector_result,
            "code_graph": graph_result
        }
    except Exception as e:
        logger.warning(f"Vector DB updated but code graph failed for {filepath}: {str(e)}")
        return {
            "status": "partial_success",
            "message": f"File added to vector DB but code graph failed: {filepath}",
            "vector_db": vector_result,
            "code_graph_error": str(e)
        }


async def _ltm_delete(
    filepath: Optional[str] = None,
    uuids: Optional[list] = None
) -> Dict[str, Any]:
    """
    Internal function: Unified delete from vector database AND code graph
    Called by Innocody engine triggers, not exposed to agents
    
    NEW: Uses unified endpoint that ensures proper synchronization:
    1. Get UUIDs from Vector DB BEFORE deletion
    2. Delete from Code Graph FIRST (with UUID verification)
    3. Delete from Vector DB LAST (only if Graph succeeded)
    
    This prevents race conditions and ensures data consistency.
    """
    clients = await get_clients()
    
    if filepath and uuids:
        return {"error": "Provide either filepath or uuids, not both"}
    
    if filepath:
        logger.info(f"🗑️ Using UNIFIED delete endpoint for: {filepath}")
        
        try:
            # ✅ NEW: Use unified delete endpoint
            result = await clients.ltm_client.proxy_request(
                "DELETE",
                "/unified/delete",  # ← NEW unified endpoint
                json_data={
                    "filepath": filepath,
                    "verify_uuids": True,
                    "force": False
                },
                retries=config.MAX_RETRIES
            )
            
            logger.info(f"✅ Unified delete successful: {result.get('message')}")
            return {
                "status": "success",
                "method": "unified",
                "message": result.get("message"),
                "vector_db_deleted": result.get("vector_db_deleted"),
                "graph_deleted": result.get("graph_deleted"),
                "uuids_verified": result.get("uuids_verified", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Unified delete failed for {filepath}: {e}")
            
            # Fallback to old method if unified endpoint fails
            logger.warning(f"⚠️ Falling back to old delete method...")
            return await _ltm_delete_fallback(filepath)

    elif uuids:
        # Delete by UUIDs (only from Vector DB)
        logger.info(f"🗑️ Deleting by UUIDs: {len(uuids)} items")
        result = await clients.ltm_client.proxy_request(
            "DELETE",
            "/vectors/uuids",
            json_data=uuids,
            retries=config.MAX_RETRIES
        )
        return result
    
    else:
        return {"error": "Must provide either filepath or uuids"}


async def _ltm_delete_fallback(filepath: str) -> Dict[str, Any]:
    """
    Fallback delete method (old approach) if unified endpoint fails
    
    ❌ WARNING: This method has race condition issues!
    Only used as emergency fallback.
    """
    clients = await get_clients()
    
    logger.warning(f"⚠️ Using FALLBACK delete (has race conditions!)")
    
    # Old approach: Vector DB first, then Graph
    vector_result = await clients.ltm_client.proxy_request(
        "DELETE",
        "/vectors/filepath",
        params={"filepath": filepath},
        retries=config.MAX_RETRIES
    )
    
    try:
        graph_result = await clients.ltm_client.proxy_request(
            "DELETE",
            "/graph/files/",
            json_data={
                "file_name": filepath,
                "file_action": "remove"
            },
            retries=config.MAX_RETRIES
        )
        
        return {
            "status": "success",
            "method": "fallback",
            "message": f"File deleted (fallback method): {filepath}",
            "vector_db": vector_result,
            "code_graph": graph_result
        }
    except Exception as e:
        logger.error(f"❌ Graph delete failed: {e}")
        return {
            "status": "partial",
            "method": "fallback",
            "message": f"Vector DB deleted, Graph failed: {filepath}",
            "vector_db": vector_result,
            "code_graph_error": str(e)
        }


async def _ltm_chunk_file(file_path: str) -> Dict[str, Any]:
    """
    Internal function: Chunk file using AST
    Called by Innocody engine triggers, not exposed to agents
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/files/chunks",
        params={"file_path": file_path},
        retries=config.MAX_RETRIES
    )


async def _ltm_build_graph(graph_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal function: Build graph from pre-parsed AST data (PRODUCTION READY)
    Called by Innocody engine triggers, not exposed to agents
    
    This is the production-ready solution that eliminates LTM's filesystem dependency.
    Engine sends complete graph payload, LTM just stores it.
    """
    clients = await get_clients()
    
    logger.info(f"🏗️ Building for: {graph_payload.get('file_path', 'unknown')}")
    
    try:
        result = await clients.ltm_client.proxy_request(
            "POST",
            "/unified/build_from_engine",
            json_data=graph_payload,
            retries=config.MAX_RETRIES
        )
        
        logger.info(f"✅ Graph built successfully: {result.get('file_uuid', 'unknown')}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Graph build failed: {str(e)}")
        raise


async def _ltm_add_file_content(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal function: Add file with content (NO filesystem dependency)
    Called by Innocody engine triggers, not exposed to agents
    
    PRODUCTION-READY: Engine sends file content, LTM doesn't need to read disk.
    Cross-platform compatible (Windows/Mac/Linux/Docker).
    
    Expected payload:
    {
        "filepath": "src/main.rs",
        "content": "fn main() { ... }",
        "language": "rust",
        "repo_name": "my-project" (optional)
    }
    """
    clients = await get_clients()
    
    logger.info(f"📄 Adding file with content: {file_data.get('filepath', 'unknown')}")
    
    try:
        result = await clients.ltm_client.proxy_request(
            "POST",
            "/vectors/add_file_content",
            json_data=file_data,
            retries=config.MAX_RETRIES
        )
        
        logger.info(f"✅ File added successfully: {result.get('file_uuid', 'unknown')}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Add file content failed: {str(e)}")
        raise


async def _ltm_update_files(file_updates: list) -> Dict[str, Any]:
    """
    Internal function: Update files in graph and vector DB
    Called by Innocody engine triggers, not exposed to agents
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "PUT",
        "/repos/files/changes/",
        json_data=file_updates,
        retries=config.MAX_RETRIES
    )


# ============================================================================
# HTTP SERVER SETUP - SAME PATTERN AS FASTMCP_SERVER
# ============================================================================

def create_http_app():
    """
    Create HTTP app using COMBINED ROUTES pattern from FastMCP docs
    Reference: https://gofastmcp.com/integrations/fastapi#offering-an-llm-friendly-api
    
    This pattern combines MCP routes and custom routes into a single app.
    SAME as fastmcp_server/server_http.py implementation.
    """
    # Create the MCP's ASGI app with '/mcp' path
    mcp_app = mcp.http_app(path='/mcp')
    
    # Create a new FastAPI app that combines MCP routes
    combined_app = FastAPI(
        title=config.AGGREGATOR_NAME,
        description="Unified MCP Aggregator - Proxies to multiple MCP servers",
        version=config.AGGREGATOR_VERSION,
        routes=[*mcp_app.routes],  # MCP routes at /mcp/*
        lifespan=mcp_app.lifespan,  # Important: use MCP lifespan
    )
    
    # ========================================================================
    # ADD ROOT ENDPOINT
    # ========================================================================
    @combined_app.get("/", operation_id="root")
    async def root():
        """Root endpoint - Server information"""
        return {
            "status": "ok",
            "service": "MCP Aggregator",
            "version": config.AGGREGATOR_VERSION,
            "name": config.AGGREGATOR_NAME,
            "endpoints": {
                "mcp": "/mcp",
                "openapi_docs": "/docs",
                "openapi_schema": "/openapi.json",
                "admin": {
                    "health": "/admin/health",
                    "info": "/admin/info",
                    "stats": "/admin/stats/{type}"
                },
                "triggers": {
                    "memory_ingest": "/triggers/memory/ingest",
                    "memory_ingest_code_changes": "/triggers/memory/ingest_code_changes",
                    "ltm_process_repo": "/triggers/ltm/process_repo",
                    "ltm_add_file": "/triggers/ltm/add_file",
                    "ltm_update_files": "/triggers/ltm/update_files",
                    "ltm_delete": "/triggers/ltm/delete",
                    "ltm_chunk_file": "/triggers/ltm/chunk_file"
                }
            },
            "connected_servers": {
                "zepai_memory_server": config.MEMORY_SERVER_URL,
                "ltm_vector_server": config.LTM_SERVER_URL
            },
            "agent_tools_count": 2,
            "agent_tools": [
                "memory_search",
                "ltm_search"
            ],
            "deprecated_tools": {
                "ltm_query_vector": "Use ltm_search instead (still supported for backward compatibility)",
                "ltm_search_file": "Removed - use ltm_search which includes graph data",
                "ltm_find_code": "Removed - use ltm_search which includes graph data"
            },
            "innocody_triggers_count": 7,
            "note": "Agents use search tools only. Innocody engine uses trigger endpoints for updates."
        }
    
    # ========================================================================
    # ADD ADMIN ENDPOINTS (HTTP only, not MCP tools)
    # ========================================================================
    @combined_app.get("/admin/health", operation_id="admin_health_check")
    async def admin_health_check():
        """Admin endpoint: Check health of all servers"""
        return await _health_check()
    
    @combined_app.get("/admin/info", operation_id="admin_server_info")
    async def admin_server_info():
        """Admin endpoint: Get detailed server information"""
        return await _get_server_info()
    
    @combined_app.get("/admin/stats/{stats_type}", operation_id="admin_stats")
    async def admin_stats(stats_type: str, project_id: Optional[str] = None):
        """Admin endpoint: Get statistics (project or cache)"""
        return await _memory_stats(stats_type, project_id)
    
    # ========================================================================
    # INNOCODY TRIGGER ENDPOINTS (HTTP only, for Innocody engine)
    # ========================================================================
    
    # Memory triggers
    @combined_app.post("/triggers/memory/ingest", operation_id="trigger_memory_ingest")
    async def trigger_memory_ingest(
        request: Request,
        content_type: str = "text",
        project_id: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Innocody trigger: Ingest content into memory"""
        # Parse body directly from request
        try:
            content = await request.json()
        except Exception as e:
            # If JSON parsing fails, try text
            content = await request.body()
            content = content.decode('utf-8') if content else ""
        
        return await _memory_ingest(content, content_type, project_id, language, metadata)
    
    @combined_app.post("/triggers/memory/ingest_code_changes", operation_id="trigger_memory_ingest_code_changes")
    async def trigger_memory_ingest_code_changes(
        payload: Dict[str, Any] = Body(...)
    ):
        """
        Innocody trigger: Ingest DiffChunks as structured code changes
        
        This endpoint properly converts DiffChunks to code change metadata
        and stores them in Neo4j with file/function entity relationships.
        
        Expected payload:
        {
            "diff_chunks": [
                {
                    "file_name": "path/to/file.py",
                    "file_action": "edit",
                    "line1": 10,
                    "line2": 20,
                    "lines_add": "new code",
                    "lines_remove": "old code",
                    ...
                }
            ],
            "project_id": "my_project",
            "chat_id": "optional_chat_id"
        }
        """
        diff_chunks = payload.get("diff_chunks", [])
        project_id = payload.get("project_id", "default_project")
        chat_id = payload.get("chat_id")
        return await _memory_ingest_code_changes(diff_chunks, project_id, chat_id)
    
    # LTM triggers
    @combined_app.post("/triggers/ltm/process_repo", operation_id="trigger_ltm_process_repo")
    async def trigger_ltm_process_repo(repo_path: str):
        """Innocody trigger: Index repository"""
        return await _ltm_process_repo(repo_path)
    
    @combined_app.post("/triggers/ltm/add_file", operation_id="trigger_ltm_add_file")
    async def trigger_ltm_add_file(filepath: str):
        """Innocody trigger: Add file to vector DB (DEPRECATED - use add_file_content)"""
        return await _ltm_add_file(filepath)
    
    @combined_app.post("/triggers/ltm/add_file_content", operation_id="trigger_ltm_add_file_content")
    async def trigger_ltm_add_file_content(file_data: Dict[str, Any] = Body(...)):
        """
        Innocody trigger: Add file with content (PRODUCTION READY)
        
        No filesystem dependency - works across Windows/Mac/Linux/Docker.
        Engine sends file content directly.
        
        Payload:
        {
            "filepath": "src/main.rs",
            "content": "fn main() { println!(\"Hello\"); }",
            "language": "rust",
            "repo_name": "my-project"
        }
        """
        return await _ltm_add_file_content(file_data)
    
    @combined_app.put("/triggers/ltm/update_files", operation_id="trigger_ltm_update_files")
    async def trigger_ltm_update_files(file_updates: list):
        """Innocody trigger: Update files in graph and vector DB"""
        return await _ltm_update_files(file_updates)
    
    @combined_app.delete("/triggers/ltm/delete", operation_id="trigger_ltm_delete")
    async def trigger_ltm_delete(filepath: Optional[str] = None, uuids: Optional[list] = None):
        """Innocody trigger: Delete from vector DB"""
        return await _ltm_delete(filepath, uuids)
    
    @combined_app.post("/triggers/ltm/chunk_file", operation_id="trigger_ltm_chunk_file")
    async def trigger_ltm_chunk_file(file_path: str):
        """Innocody trigger: Chunk file using AST"""
        return await _ltm_chunk_file(file_path)
    
    @combined_app.post("/triggers/ltm/build_graph", operation_id="trigger_ltm_build_graph")
    async def trigger_ltm_build_graph(graph_payload: Dict[str, Any] = Body(...)):
        """
        Innocody trigger: Build graph from pre-parsed AST data (PRODUCTION READY)
        
        This endpoint receives complete graph payload from Engine, eliminating
        the need for LTM to read files from local filesystem.
        
        Payload structure:
        {
            "file_path": "/path/to/file.py",
            "relative_path": "src/main.py",
            "language": "python",
            "repo_name": "my-project",
            "functions": [...],
            "classes": [...],
            "variables": [...],
            "imports": [...]
        }
        """
        return await _ltm_build_graph(graph_payload)
    
    # ========================================================================
    # MCP TOOL CALL ENDPOINT (For Innocody Engine tool integration)
    # ========================================================================
    from pydantic import BaseModel
    
    class McpToolCallRequest(BaseModel):
        name: str
        arguments: Dict[str, Any]
    
    @combined_app.post("/mcp/tools/call", operation_id="mcp_tool_call")
    async def mcp_tool_call(request: McpToolCallRequest):
        """
        Call MCP tools from Innocody Engine
        Maps tool names to actual MCP tool functions
        """
        tool_name = request.name
        args = request.arguments
        
        try:
            # Map tool names to internal functions
            if tool_name == "memory_search":
                result = await _memory_search(
                    query=args.get("query", ""),
                    project_id=args.get("project_id"),
                    limit=args.get("limit", 10),
                    use_llm_classification=args.get("use_llm_classification", False)
                )
            elif tool_name == "ltm_search" or tool_name == "ltm_query_vector":
                # Support both new name (ltm_search) and legacy name (ltm_query_vector)
                result = await _ltm_query_vector(
                    query=args.get("query", ""),
                    top_k=args.get("top_k", 10)
                )
            else:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": f"Unknown tool: {tool_name}",
                        "available_tools": ["memory_search", "ltm_search"],
                        "note": "ltm_search provides unified vector + graph search. Old tools (ltm_search_file, ltm_find_code) are deprecated."
                    }
                )
            
            # Return in MCP tool response format
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result) if not isinstance(result, (dict, list)) else None,
                        "data": result if isinstance(result, (dict, list)) else None
                    }
                ],
                "isError": False
            }
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}"
                    }
                ],
                "isError": True
            }
    
    # ========================================================================
    # ADD CORS MIDDLEWARE
    # ========================================================================
    combined_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware added")
    
    # ========================================================================
    # ADD EXCEPTION HANDLERS
    # ========================================================================
    @combined_app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        logger.error(f"HTTP {exc.status_code} error: {exc.detail} - Path: {request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": str(request.url.path),
                "status_code": exc.status_code
            }
        )
    
    @combined_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors"""
        logger.error(f"Validation error: {exc.errors()} - Path: {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": exc.errors(),
                "path": str(request.url.path)
            }
        )
    
    @combined_app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions"""
        logger.exception(f"Unhandled exception: {str(exc)} - Path: {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": str(exc),
                "path": str(request.url.path),
                "type": type(exc).__name__
            }
        )
    
    logger.info(f"Combined MCP routes")
    logger.info(f"Total routes: {len(combined_app.routes)}")
    
    return combined_app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("MCP Aggregator Server - Unified Knowledge Interface")
    logger.info("=" * 70)
    logger.info(f"Server: {config.AGGREGATOR_NAME} v{config.AGGREGATOR_VERSION}")
    logger.info(f"")
    logger.info(f"Connected Servers:")
    logger.info(f"  ✅ ZepAI Memory Server: {config.MEMORY_SERVER_URL}")
    logger.info(f"  ✅ LTM Vector Server: {config.LTM_SERVER_URL}")
    logger.info(f"")
    logger.info(f"🤖 MCP Tools for Agents: 2 unified search tools")
    logger.info(f"  - memory_search: Search in conversation/memory knowledge graph (ZepAI)")
    logger.info(f"  - ltm_search: Unified vector + graph code search (LTM)")
    logger.info(f"    └─ Returns both vector results AND graph relationships in one call")
    logger.info(f"    └─ Deprecated: ltm_query_vector, ltm_search_file, ltm_find_code")
    logger.info(f"")
    logger.info(f"🔧 Innocody Trigger Endpoints: 8 endpoints")
    logger.info(f"  Memory:")
    logger.info(f"    POST /triggers/memory/ingest")
    logger.info(f"    POST /triggers/memory/ingest_code_changes")
    logger.info(f"  LTM:")
    logger.info(f"    POST /triggers/ltm/process_repo")
    logger.info(f"    POST /triggers/ltm/add_file")
    logger.info(f"    PUT  /triggers/ltm/update_files")
    logger.info(f"    DELETE /triggers/ltm/delete")
    logger.info(f"    POST /triggers/ltm/chunk_file")
    logger.info(f"    POST /triggers/ltm/build_graph (NEW - PRODUCTION READY) 🏗️")
    logger.info(f"")
    logger.info(f"⚙️  Admin Endpoints: 3 endpoints")
    logger.info(f"    GET /admin/health")
    logger.info(f"    GET /admin/info")
    logger.info(f"    GET /admin/stats/{{type}}")
    logger.info(f"")
    logger.info(f"HTTP Endpoints:")
    logger.info(f"  - MCP Endpoint: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/mcp")
    logger.info(f"  - OpenAPI Docs: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/docs")
    logger.info(f"  - Root API: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/")
    logger.info(f"")
    logger.info(f"To use:")
    logger.info(f"  1. View docs: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/docs")
    logger.info(f"  2. Connect MCP clients to: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/mcp")
    logger.info(f"     (MCP protocol handles SSE/messages automatically)")
    logger.info(f"  3. Run tests: python test_final.py")
    logger.info("=" * 70)
    
    # Create HTTP app
    http_app = create_http_app()
    
    logger.info(f"Starting HTTP MCP server on port {config.AGGREGATOR_PORT}...")
    logger.info("Press Ctrl+C to stop")
    
    # Run with uvicorn with optimized configuration - SAME AS FASTMCP_SERVER
    try:
        uvicorn.run(
            http_app,
            host=config.AGGREGATOR_HOST,
            port=config.AGGREGATOR_PORT,
            log_level="info",
            access_log=True,
            use_colors=True,
            limit_concurrency=100,
            timeout_keep_alive=5,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(close_clients())
