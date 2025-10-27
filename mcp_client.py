"""
HTTP Client for connecting to MCP Servers
"""
import httpx
import logging
import asyncio
from typing import Dict, Any, Optional, List
from config import config

logger = logging.getLogger(__name__)


class MCPServerClient:
    """HTTP client for MCP Server backend"""
    
    def __init__(self, server_name: str, base_url: str, timeout: int = 30):
        self.server_name = server_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )
        self.is_healthy = True
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = await self.client.get(
                f"{self.base_url}/",
                timeout=httpx.Timeout(5)
            )
            self.is_healthy = response.status_code == 200
            logger.info(f"{self.server_name} health check: {'✓ Healthy' if self.is_healthy else '✗ Unhealthy'}")
            return self.is_healthy
        except Exception as e:
            self.is_healthy = False
            logger.warning(f"{self.server_name} health check failed: {str(e)}")
            return False
    
    async def get_openapi_schema(self) -> Optional[Dict[str, Any]]:
        """Get OpenAPI schema from server"""
        try:
            response = await self.client.get(f"{self.base_url}/openapi.json")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get OpenAPI schema from {self.server_name}: {str(e)}")
            return None
    
    async def proxy_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Proxy HTTP request to backend server with retry logic
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., "/search")
            params: Query parameters
            json_data: JSON body
            retries: Number of retries on failure
        
        Returns:
            Response JSON or error dict
        """
        url = f"{self.base_url}{path}"
        
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    response = await self.client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await self.client.post(url, json=json_data, params=params)
                elif method.upper() == "PUT":
                    response = await self.client.put(url, json=json_data, params=params)
                elif method.upper() == "DELETE":
                    response = await self.client.delete(url, params=params)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}
                
                response.raise_for_status()
                return response.json()
            
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{retries}: {self.server_name} "
                    f"{method} {path} returned {e.response.status_code}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(config.RETRY_DELAY ** attempt)
            
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{retries}: {self.server_name} "
                    f"{method} {path} failed: {str(e)}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(config.RETRY_DELAY ** attempt)
        
        return {
            "error": f"Failed to reach {self.server_name} after {retries} attempts",
            "server": self.server_name,
            "path": path,
            "method": method
        }


class AggregatorClients:
    """Manager for multiple MCP server clients"""
    
    def __init__(self):
        # ZepAI Memory Server (Port 8002) - Knowledge Graph + Conversation Memory
        self.memory_client = MCPServerClient(
            "ZepAI Memory Server",
            config.MEMORY_SERVER_URL,
            config.MEMORY_SERVER_TIMEOUT
        )
        
        # LTM Vector Server (Port 8000) - Vector Database + Code Indexing
        self.ltm_client = MCPServerClient(
            "LTM Vector Server",
            config.LTM_SERVER_URL,
            config.LTM_SERVER_TIMEOUT
        )
        
        # Legacy alias for backward compatibility
        self.graph_client = self.ltm_client
    
    async def close_all(self):
        """Close all clients"""
        await self.memory_client.close()
        await self.ltm_client.close()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all servers"""
        return {
            "memory_server": await self.memory_client.health_check(),
            "ltm_server": await self.ltm_client.health_check(),
        }
    
    async def get_all_schemas(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get OpenAPI schemas from all servers"""
        return {
            "memory_server": await self.memory_client.get_openapi_schema(),
            "ltm_server": await self.ltm_client.get_openapi_schema(),
        }


# Singleton instance
_clients: Optional[AggregatorClients] = None


async def get_clients() -> AggregatorClients:
    """Get or create aggregator clients instance"""
    global _clients
    if _clients is None:
        _clients = AggregatorClients()
    return _clients


async def close_clients():
    """Close all clients"""
    global _clients
    if _clients is not None:
        await _clients.close_all()
        _clients = None
