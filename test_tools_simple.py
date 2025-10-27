"""
Simple Tool Testing via REST API
=================================
Test tools by calling backend REST APIs directly through aggregator.

Run:
    python test_tools_simple.py
"""
import asyncio
import httpx
from datetime import datetime
import json

# Server URLs
AGGREGATOR_URL = "http://localhost:8003"
ZEPAI_URL = "http://localhost:8002"
LTM_URL = "http://localhost:8000"

# Colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_test(test_name: str):
    print(f"{Colors.YELLOW}▶ {test_name}{Colors.RESET}")

def print_success(message: str):
    print(f"  {Colors.GREEN}✓ {message}{Colors.RESET}")

def print_error(message: str):
    print(f"  {Colors.RED}✗ {message}{Colors.RESET}")

def print_info(message: str):
    print(f"  {Colors.BLUE}ℹ {message}{Colors.RESET}")


class SimpleTester:
    """Simple REST API testing"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.passed = 0
        self.failed = 0
    
    async def close(self):
        await self.client.aclose()
    
    async def test_aggregator_info(self):
        """Test aggregator root endpoint"""
        print_test("Test: Aggregator Info")
        try:
            response = await self.client.get(f"{AGGREGATOR_URL}/")
            data = response.json()
            
            print_success(f"Aggregator is running")
            print_info(f"Service: {data['name']}")
            print_info(f"Version: {data['version']}")
            print_info(f"Tools: {data['tools_count']}")
            print_info(f"Breakdown: {data['tools_breakdown']}")
            self.passed += 1
        except Exception as e:
            print_error(f"Failed: {e}")
            self.failed += 1
    
    async def test_openapi_docs(self):
        """Test OpenAPI documentation"""
        print_test("Test: OpenAPI Documentation")
        try:
            response = await self.client.get(f"{AGGREGATOR_URL}/openapi.json")
            data = response.json()
            
            paths = data.get("paths", {})
            print_success(f"OpenAPI docs available")
            print_info(f"API paths: {len(paths)}")
            self.passed += 1
        except Exception as e:
            print_error(f"Failed: {e}")
            self.failed += 1
    
    async def test_backend_connectivity(self):
        """Test backend server connectivity"""
        print_test("Test: Backend Connectivity")
        
        # Test ZepAI
        try:
            response = await self.client.get(f"{ZEPAI_URL}/")
            if response.status_code == 200:
                print_success("ZepAI backend accessible")
                self.passed += 1
            else:
                print_error(f"ZepAI returned {response.status_code}")
                self.failed += 1
        except Exception as e:
            print_error(f"ZepAI failed: {e}")
            self.failed += 1
        
        # Test LTM
        try:
            response = await self.client.get(f"{LTM_URL}/docs")
            if response.status_code == 200:
                print_success("LTM backend accessible")
                self.passed += 1
            else:
                print_error(f"LTM returned {response.status_code}")
                self.failed += 1
        except Exception as e:
            print_error(f"LTM failed: {e}")
            self.failed += 1
    
    async def test_mcp_endpoint(self):
        """Test MCP SSE endpoint"""
        print_test("Test: MCP SSE Endpoint")
        try:
            # FastMCP exposes /mcp as base path, check if it exists
            # Try different common MCP endpoints
            endpoints_to_try = [
                "/mcp",
                "/mcp/sse", 
                "/mcp/messages"
            ]
            
            found = False
            for endpoint in endpoints_to_try:
                try:
                    response = await self.client.get(
                        f"{AGGREGATOR_URL}{endpoint}",
                        headers={"Accept": "text/event-stream"},
                        timeout=2.0
                    )
                    
                    # Any response (200, 400, 405) means endpoint exists
                    if response.status_code in [200, 400, 405]:
                        print_success(f"MCP endpoint exists: {endpoint}")
                        print_info(f"Status: {response.status_code}")
                        found = True
                        self.passed += 1
                        break
                except httpx.TimeoutException:
                    # Timeout means SSE connection opened (good!)
                    print_success(f"MCP SSE endpoint exists: {endpoint} (timeout expected)")
                    found = True
                    self.passed += 1
                    break
                except:
                    continue
            
            if not found:
                print_error("No MCP endpoint found")
                print_info("Note: MCP tools still work via aggregator")
                self.failed += 1
                
        except Exception as e:
            print_error(f"Failed: {e}")
            self.failed += 1
    
    async def test_tool_count(self):
        """Verify tool count"""
        print_test("Test: Tool Count Verification")
        try:
            response = await self.client.get(f"{AGGREGATOR_URL}/")
            data = response.json()
            
            expected = 12
            actual = data['tools_count']
            
            if actual == expected:
                print_success(f"Tool count correct: {actual}")
                print_info(f"Health: {data['tools_breakdown']['health']}")
                print_info(f"Memory: {data['tools_breakdown']['memory']}")
                print_info(f"Vector: {data['tools_breakdown']['vector']}")
                self.passed += 1
            else:
                print_error(f"Expected {expected}, got {actual}")
                self.failed += 1
        except Exception as e:
            print_error(f"Failed: {e}")
            self.failed += 1
    
    async def test_consolidated_tools(self):
        """Verify consolidated tools"""
        print_test("Test: Consolidated Tools")
        try:
            response = await self.client.get(f"{AGGREGATOR_URL}/")
            data = response.json()
            
            note = data.get('note', '')
            if 'consolidated' in note.lower():
                print_success("Tools are consolidated")
                print_info(f"Note: {note}")
                self.passed += 1
            else:
                print_error("Consolidation note missing")
                self.failed += 1
        except Exception as e:
            print_error(f"Failed: {e}")
            self.failed += 1
    
    async def run_all_tests(self):
        """Run all tests"""
        print_header("SIMPLE TOOL TESTING")
        print(f"Testing aggregator and backend connectivity")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print_header("AGGREGATOR TESTS")
        await self.test_aggregator_info()
        await self.test_openapi_docs()
        await self.test_tool_count()
        await self.test_consolidated_tools()
        await self.test_mcp_endpoint()
        
        print_header("BACKEND TESTS")
        await self.test_backend_connectivity()
        
        # Summary
        print_header("TEST SUMMARY")
        total = self.passed + self.failed
        print(f"Total tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}")
        
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n{Colors.YELLOW}Note:{Colors.RESET}")
        print(f"  This tests aggregator infrastructure and connectivity.")
        print(f"  For full MCP protocol testing, use MCP-compatible client.")
        print(f"  MCP endpoint: {AGGREGATOR_URL}/mcp/sse (SSE transport)")


async def main():
    """Main test runner"""
    tester = SimpleTester()
    try:
        await tester.run_all_tests()
    finally:
        await tester.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Test failed: {str(e)}{Colors.RESET}")
