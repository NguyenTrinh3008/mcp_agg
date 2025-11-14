"""
Quick test for memory_search tool via MCP Aggregator
"""
import requests
import json

# Test memory_search tool
url = "http://localhost:9003/mcp/tools/call"
payload = {
    "name": "memory_search",
    "arguments": {
        "query": "pacman game",
        "limit": 5
    }
}

print("🧪 Testing memory_search tool...")
print(f"Request: {json.dumps(payload, indent=2)}\n")

response = requests.post(url, json=payload)

print(f"Status: {response.status_code}")
print(f"Response:\n{json.dumps(response.json(), indent=2)}")

if response.status_code == 200:
    print("\n✅ Memory search successful! ZepAI is receiving requests.")
else:
    print(f"\n❌ Error: {response.status_code}")
