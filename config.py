"""
MCP Aggregator Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for MCP Aggregator"""
    
    # Aggregator Server settings
    AGGREGATOR_HOST = os.getenv("AGGREGATOR_HOST", "0.0.0.0")
    AGGREGATOR_PORT = int(os.getenv("AGGREGATOR_PORT", "9003"))
    AGGREGATOR_NAME = "Unified Knowledge Server"
    AGGREGATOR_VERSION = "1.0.0"
    
    # Memory Server (ZepAI FastMCP Server - Knowledge Graph)
    MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:9001")
    MEMORY_SERVER_TIMEOUT = int(os.getenv("MEMORY_SERVER_TIMEOUT", "30"))
    
    # LTM Server (Vector Database - Code Indexing)
    LTM_SERVER_URL = os.getenv("LTM_SERVER_URL", "http://localhost:9000")
    LTM_SERVER_TIMEOUT = int(os.getenv("LTM_SERVER_TIMEOUT", "30"))
    
    # Legacy: Keep for backward compatibility
    GRAPH_SERVER_URL = os.getenv("GRAPH_SERVER_URL", os.getenv("LTM_SERVER_URL", "http://localhost:9000"))
    GRAPH_SERVER_TIMEOUT = int(os.getenv("GRAPH_SERVER_TIMEOUT", os.getenv("LTM_SERVER_TIMEOUT", "30")))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))  # seconds
    
    # Health check
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds


config = Config()
