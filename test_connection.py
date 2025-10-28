"""
Test MCP Aggregator Connection to Both Servers
"""
import asyncio
import httpx
import sys

# Server URLs
AGGREGATOR_URL = "http://localhost:8003"
MEMORY_SERVER_URL = "http://localhost:8002"
LTM_SERVER_URL = "http://localhost:8000"


async def test_server_health(url: str, name: str) -> bool:
    """Test if a server is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/")
            if response.status_code == 200:
                print(f"✅ {name} is healthy")
                data = response.json()
                print(f"   Response: {data}")
                return True
            else:
                print(f"❌ {name} returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ {name} connection failed: {str(e)}")
        return False


async def test_aggregator_tools():
    """Test aggregator tools endpoint"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get server info
            response = await client.get(f"{AGGREGATOR_URL}/")
            if response.status_code == 200:
                data = response.json()
                print(f"\n📊 Aggregator Info:")
                print(f"   Service: {data.get('service')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Tools Count: {data.get('tools_count')}")
                print(f"   Tools Breakdown: {data.get('tools_breakdown')}")
                print(f"   Connected Servers:")
                for server, url in data.get('connected_servers', {}).items():
                    print(f"     - {server}: {url}")
                return True
            else:
                print(f"❌ Failed to get aggregator info")
                return False
    except Exception as e:
        print(f"❌ Aggregator tools test failed: {str(e)}")
        return False


async def test_ltm_endpoints():
    """Test LTM server endpoints"""
    print(f"\n🔍 Testing LTM Server Endpoints:")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test OpenAPI schema
            response = await client.get(f"{LTM_SERVER_URL}/openapi.json")
            if response.status_code == 200:
                schema = response.json()
                paths = schema.get('paths', {})
                print(f"   ✅ OpenAPI schema available")
                print(f"   📝 Available endpoints ({len(paths)}):")
                
                # Group by category
                repos_endpoints = [p for p in paths.keys() if '/repos' in p]
                graph_endpoints = [p for p in paths.keys() if '/graph' in p]
                vector_endpoints = [p for p in paths.keys() if '/vectors' in p]
                
                if repos_endpoints:
                    print(f"      Repos: {', '.join(repos_endpoints)}")
                if graph_endpoints:
                    print(f"      Graph: {', '.join(graph_endpoints)}")
                if vector_endpoints:
                    print(f"      Vectors: {', '.join(vector_endpoints)}")
                
                return True
            else:
                print(f"   ❌ Failed to get OpenAPI schema")
                return False
    except Exception as e:
        print(f"   ❌ LTM endpoints test failed: {str(e)}")
        return False


async def test_memory_endpoints():
    """Test Memory server endpoints"""
    print(f"\n🔍 Testing Memory Server Endpoints:")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test OpenAPI schema
            response = await client.get(f"{MEMORY_SERVER_URL}/openapi.json")
            if response.status_code == 200:
                schema = response.json()
                paths = schema.get('paths', {})
                print(f"   ✅ OpenAPI schema available")
                print(f"   📝 Available endpoints ({len(paths)}):")
                
                # Group by category
                search_endpoints = [p for p in paths.keys() if '/search' in p]
                ingest_endpoints = [p for p in paths.keys() if '/ingest' in p or '/conversation' in p]
                admin_endpoints = [p for p in paths.keys() if '/stats' in p or '/cache' in p]
                
                if search_endpoints:
                    print(f"      Search: {', '.join(search_endpoints)}")
                if ingest_endpoints:
                    print(f"      Ingest: {', '.join(ingest_endpoints)}")
                if admin_endpoints:
                    print(f"      Admin: {', '.join(admin_endpoints)}")
                
                return True
            else:
                print(f"   ❌ Failed to get OpenAPI schema")
                return False
    except Exception as e:
        print(f"   ❌ Memory endpoints test failed: {str(e)}")
        return False


async def main():
    """Main test function"""
    print("=" * 70)
    print("MCP Aggregator Connection Test")
    print("=" * 70)
    
    # Test individual servers
    print("\n🏥 Testing Server Health:")
    memory_ok = await test_server_health(MEMORY_SERVER_URL, "Memory Server (8002)")
    ltm_ok = await test_server_health(LTM_SERVER_URL, "LTM Server (8000)")
    aggregator_ok = await test_server_health(AGGREGATOR_URL, "Aggregator (8003)")
    
    if not memory_ok or not ltm_ok:
        print("\n❌ Backend servers are not running!")
        print("Please start:")
        if not memory_ok:
            print(f"  1. Memory Server: cd ZepAI/fastmcp_server && python server_http.py")
        if not ltm_ok:
            print(f"  2. LTM Server: cd LTM && uvicorn main:app --port 8000")
        sys.exit(1)
    
    if not aggregator_ok:
        print("\n❌ Aggregator is not running!")
        print("Please start: python aggregator_server.py")
        sys.exit(1)
    
    # Test aggregator
    await test_aggregator_tools()
    
    # Test backend endpoints
    await test_memory_endpoints()
    await test_ltm_endpoints()
    
    print("\n" + "=" * 70)
    print("✅ All connection tests passed!")
    print("=" * 70)
    print("\n🤖 MCP Tools for Agents: 4 search tools")
    print("   - memory_search: Search in memory knowledge graph")
    print("   - ltm_query_vector: Semantic code search")
    print("   - ltm_search_file: Get all chunks of a file")
    print("   - ltm_find_code: Find code entities in graph")
    print("\n🔧 Innocody Trigger Endpoints: 6 endpoints")
    print("   Memory: POST /triggers/memory/ingest")
    print("   LTM: POST /triggers/ltm/process_repo, add_file, update_files, delete, chunk_file")
    print("\n⚙️  Admin Endpoints: 3 endpoints")
    print("   GET /admin/health, /admin/info, /admin/stats/{type}")
    print("\n🚀 Ready to use! Access docs at: http://localhost:8003/docs")


if __name__ == "__main__":
    asyncio.run(main())
