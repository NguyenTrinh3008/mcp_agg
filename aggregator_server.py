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
from typing import Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, status
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

# ============================================================================
# INTERNAL ADMIN FUNCTIONS (Not exposed as MCP tools)
# ============================================================================

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
        "conversation": "/conversation/ingest"
    }
    
    if content_type not in endpoint_map:
        return {
            "error": f"Invalid content_type: {content_type}",
            "valid_types": list(endpoint_map.keys())
        }
    
    # Build payload based on content type
    if content_type == "text":
        payload = {
            "text": content,
            "metadata": metadata or {}
        }
    elif content_type == "code":
        payload = {
            "code": content,
            "metadata": metadata or {}
        }
        if language:
            payload["language"] = language
    elif content_type == "json":
        payload = {
            "data": content,
            "metadata": metadata or {}
        }
    elif content_type == "conversation":
        payload = content.copy() if isinstance(content, dict) else {}
    
    if project_id:
        payload["project_id"] = project_id
    
    return await clients.memory_client.proxy_request(
        "POST",
        endpoint_map[content_type],
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
# AGENT TOOLS - LTM SEARCH/QUERY (Exposed as MCP tools)
# ============================================================================

@mcp.tool()
async def ltm_query_vector(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search code semantically using vector embeddings
    
    Args:
        query: Natural language query to find similar code
        top_k: Number of top results to return
    
    Returns:
        Most similar code chunks from vector database
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "GET",
        "/vectors/query",
        params={"query": query, "top_k": top_k},
        retries=config.MAX_RETRIES
    )


@mcp.tool()
async def ltm_search_file(filepath: str) -> Dict[str, Any]:
    """
    Get all indexed chunks of a specific file
    
    Args:
        filepath: Absolute path to file
    
    Returns:
        All code chunks from the file in vector database
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "GET",
        "/vectors/files",
        params={"filepath": filepath},
        retries=config.MAX_RETRIES
    )


@mcp.tool()
async def ltm_find_code(query: str) -> Dict[str, Any]:
    """
    Find code entities (functions, classes) in knowledge graph
    
    Args:
        query: Search query to find code entities
    
    Returns:
        Functions, classes, variables matching the query from graph
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/graph/find_code/",
        params={"query": query},
        retries=config.MAX_RETRIES
    )


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
    Internal function: Add file to vector database
    Called by Innocody engine triggers, not exposed to agents
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/vectors/files",
        params={"filepath": filepath},
        retries=config.MAX_RETRIES
    )


async def _ltm_delete(
    filepath: Optional[str] = None,
    uuids: Optional[list] = None
) -> Dict[str, Any]:
    """
    Internal function: Delete from vector database
    Called by Innocody engine triggers, not exposed to agents
    """
    clients = await get_clients()
    
    if filepath and uuids:
        return {"error": "Provide either filepath or uuids, not both"}
    
    if filepath:
        return await clients.ltm_client.proxy_request(
            "DELETE",
            "/vectors/filepath",
            params={"filepath": filepath},
            retries=config.MAX_RETRIES
        )
    elif uuids:
        return await clients.ltm_client.proxy_request(
            "DELETE",
            "/vectors/uuids",
            json_data=uuids,
            retries=config.MAX_RETRIES
        )
    else:
        return {"error": "Must provide either filepath or uuids"}


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
            "agent_tools_count": 4,
            "agent_tools": [
                "memory_search",
                "ltm_query_vector",
                "ltm_search_file",
                "ltm_find_code"
            ],
            "innocody_triggers_count": 6,
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
        content: str | Dict[str, Any],
        content_type: str = "text",
        project_id: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Innocody trigger: Ingest content into memory"""
        return await _memory_ingest(content, content_type, project_id, language, metadata)
    
    # LTM triggers
    @combined_app.post("/triggers/ltm/process_repo", operation_id="trigger_ltm_process_repo")
    async def trigger_ltm_process_repo(repo_path: str):
        """Innocody trigger: Index repository"""
        return await _ltm_process_repo(repo_path)
    
    @combined_app.post("/triggers/ltm/add_file", operation_id="trigger_ltm_add_file")
    async def trigger_ltm_add_file(filepath: str):
        """Innocody trigger: Add file to vector DB"""
        return await _ltm_add_file(filepath)
    
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
    logger.info(f"🤖 MCP Tools for Agents: 4 search tools")
    logger.info(f"  - memory_search: Search in memory knowledge graph")
    logger.info(f"  - ltm_query_vector: Semantic code search (vector DB)")
    logger.info(f"  - ltm_search_file: Get all chunks of a file")
    logger.info(f"  - ltm_find_code: Find code entities in graph")
    logger.info(f"")
    logger.info(f"🔧 Innocody Trigger Endpoints: 6 endpoints")
    logger.info(f"  Memory:")
    logger.info(f"    POST /triggers/memory/ingest")
    logger.info(f"  LTM:")
    logger.info(f"    POST /triggers/ltm/process_repo")
    logger.info(f"    POST /triggers/ltm/add_file")
    logger.info(f"    PUT  /triggers/ltm/update_files")
    logger.info(f"    DELETE /triggers/ltm/delete")
    logger.info(f"    POST /triggers/ltm/chunk_file")
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
