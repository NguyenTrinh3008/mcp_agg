"""
MCP Aggregator Server
=====================
Unified MCP interface that proxies requests to multiple backend MCP servers.

Architecture:
- Aggregator Server (Port 8003) - This server
- ZepAI Memory Server (Port 8002) - Knowledge Graph + Conversation Memory
- LTM Vector Server (Port 8000) - Vector Database + Code Indexing

Current Status:
    ZepAI Memory Server (8002) - READY (4 tools)
    LTM Vector Server (8000) - READY (6 tools)
    Total: 12 tools available (consolidated from 17)

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
logger.info("Status: ZepAI (8002) ✅ | LTM (8000) ✅ | Total: 12 tools (consolidated)")

# ============================================================================
# HEALTH & STATUS TOOLS
# ============================================================================

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Check health status of all connected servers
    
    Returns:
        Dictionary with health status of each server
    """
    clients = await get_clients()
    health_status = await clients.health_check_all()
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "servers": health_status,
        "aggregator": "healthy"
    }


@mcp.tool()
async def get_server_info() -> Dict[str, Any]:
    """
    Get information about all connected servers and available tools
    
    Returns:
        Dictionary with server information and available endpoints
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
            "tools": 4,
            "description": "Knowledge Graph + Conversation Memory (consolidated)"
        }
    
    # LTM Vector Server info
    if schemas.get("ltm_server"):
        ltm_schema = schemas["ltm_server"]
        info["connected_servers"]["ltm_vector_server"] = {
            "url": config.LTM_SERVER_URL,
            "title": ltm_schema.get("info", {}).get("title", "LTM Vector Database"),
            "version": ltm_schema.get("info", {}).get("version", "Unknown"),
            "endpoints_count": len(ltm_schema.get("paths", {})),
            "tools": 6,
            "description": "Vector Database + Code Indexing (consolidated)"
        }
    
    return info


# ============================================================================
# MEMORY SERVER TOOLS (Proxy to Port 8002)
# ============================================================================

@mcp.tool()
async def memory_search(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 10,
    use_llm_classification: bool = False
) -> Dict[str, Any]:
    """
    Search in memory knowledge graph
    
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


@mcp.tool()
async def memory_search_code(
    query: str,
    project_id: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search code memories with filters
    
    Args:
        query: Search query string
        project_id: Project ID (optional)
        limit: Maximum number of results
    
    Returns:
        Code search results from memory server
    """
    clients = await get_clients()
    
    payload = {
        "query": query,
        "limit": limit
    }
    if project_id:
        payload["project_id"] = project_id
    
    return await clients.memory_client.proxy_request(
        "POST",
        "/search/code",
        json_data=payload,
        retries=config.MAX_RETRIES
    )


@mcp.tool()
async def memory_ingest(
    content: str | Dict[str, Any],
    content_type: str = "text",
    project_id: Optional[str] = None,
    language: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Universal ingest tool for all content types
    
    Args:
        content: Content to ingest (string for text/code, dict for json/conversation)
        content_type: Type of content - "text", "code", "json", or "conversation"
        project_id: Project ID (optional)
        language: Programming language (required for code type)
        metadata: Additional metadata (optional)
    
    Returns:
        Ingestion result from memory server
    
    Examples:
        # Ingest text
        memory_ingest("Hello world", "text", project_id="proj1")
        
        # Ingest code
        memory_ingest("def hello(): pass", "code", language="python")
        
        # Ingest JSON
        memory_ingest({"key": "value"}, "json")
        
        # Ingest conversation
        memory_ingest({"messages": [...]}, "conversation")
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


@mcp.tool()
async def memory_stats(
    stats_type: str = "project",
    project_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get statistics from memory server
    
    Args:
        stats_type: Type of stats - "project" or "cache"
        project_id: Project ID (required for project stats)
    
    Returns:
        Statistics from memory server
    
    Examples:
        # Get project stats
        memory_stats("project", project_id="proj1")
        
        # Get cache stats
        memory_stats("cache")
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
# LTM VECTOR DB TOOLS (Proxy to Port 8000)
# ============================================================================

@mcp.tool()
async def ltm_process_repo(repo_path: str) -> Dict[str, Any]:
    """
    Process repository for vector indexing
    
    Args:
        repo_path: Path to repository to process
    
    Returns:
        Processing result from LTM server
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/repos/process",
        params={"repo_path": repo_path},
        retries=config.MAX_RETRIES
    )


@mcp.tool()
async def ltm_query_vector(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Query vector database for semantic code search
    
    Args:
        query: Search query string
        top_k: Number of top results to return
    
    Returns:
        Vector search results from LTM server
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
    Search for specific file in vector database
    
    Args:
        filepath: Path to file to search
    
    Returns:
        File search results from LTM server
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "GET",
        "/vectors/files",
        params={"filepath": filepath},
        retries=config.MAX_RETRIES
    )


@mcp.tool()
async def ltm_add_file(filepath: str) -> Dict[str, Any]:
    """
    Add file to vector database
    
    Args:
        filepath: Path to file to add
    
    Returns:
        Addition result from LTM server
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/vectors/files",
        params={"filepath": filepath},
        retries=config.MAX_RETRIES
    )


@mcp.tool()
async def ltm_delete(
    filepath: Optional[str] = None,
    uuids: Optional[list] = None
) -> Dict[str, Any]:
    """
    Delete from vector database by filepath or UUIDs
    
    Args:
        filepath: Path to file to delete (optional)
        uuids: List of UUIDs to delete (optional)
    
    Returns:
        Deletion result from LTM server
    
    Examples:
        # Delete by filepath
        ltm_delete(filepath="/path/to/file.py")
        
        # Delete by UUIDs
        ltm_delete(uuids=["uuid1", "uuid2"])
    
    Note:
        Must provide either filepath or uuids, not both
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


@mcp.tool()
async def ltm_chunk_file(file_path: str) -> Dict[str, Any]:
    """
    Chunk file using AST-based chunking
    
    Args:
        file_path: Path to file to chunk
    
    Returns:
        Chunking result from LTM server
    """
    clients = await get_clients()
    
    return await clients.ltm_client.proxy_request(
        "POST",
        "/files/chunks",
        params={"file_path": file_path},
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
            },
            "connected_servers": {
                "zepai_memory_server": config.MEMORY_SERVER_URL,
                "ltm_vector_server": config.LTM_SERVER_URL
            },
            "tools_count": 12,
            "tools_breakdown": {
                "health": 2,
                "memory": 4,
                "vector": 6
            },
            "note": "Consolidated from 17 tools for better usability"
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
    logger.info(f"     - Knowledge Graph + Conversation Memory")
    logger.info(f"     - 10 tools available")
    logger.info(f"")
    logger.info(f"  ✅ LTM Vector Server: {config.LTM_SERVER_URL}")
    logger.info(f"     - Vector Database + Code Indexing")
    logger.info(f"     - 7 tools available")
    logger.info(f"")
    logger.info(f"HTTP Endpoints:")
    logger.info(f"  - MCP Endpoint: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/mcp")
    logger.info(f"  - OpenAPI Docs: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/docs")
    logger.info(f"  - Root API: http://{config.AGGREGATOR_HOST}:{config.AGGREGATOR_PORT}/")
    logger.info(f"")
    logger.info(f"MCP Tools: 12 tools available (2 health + 4 memory + 6 vector)")
    logger.info(f"  Note: Consolidated from 17 tools for better usability")
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
