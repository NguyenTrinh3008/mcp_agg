"""
Test script for code changes ingestion
Verify that DiffChunks are properly ingested into Neo4j
"""

import requests
import json
from datetime import datetime

# MCP Aggregator endpoint
AGGREGATOR_URL = "http://localhost:9003"

def test_ingest_code_changes():
    """Test ingesting DiffChunks via the new endpoint"""
    
    print("=" * 70)
    print("TEST: Ingest Code Changes to Neo4j")
    print("=" * 70)
    
    # Sample DiffChunk (simulating agent creating a new file)
    payload = {
        "diff_chunks": [
            {
                "file_name": "C:/Users/Lenovo/Desktop/test_file.py",
                "file_action": "add",
                "line1": 1,
                "line2": 10,
                "lines_add": "def hello_world():\n    print('Hello World')\n\nif __name__ == '__main__':\n    hello_world()",
                "lines_remove": "",
                "file_name_rename": None,
                "application_details": "Created new test file",
                "is_file": True
            },
            {
                "file_name": "C:/Users/Lenovo/Desktop/test_file.py",
                "file_action": "edit",
                "line1": 5,
                "line2": 7,
                "lines_add": "    print('Updated message')",
                "lines_remove": "    print('Hello World')",
                "file_name_rename": None,
                "application_details": "Updated print message",
                "is_file": True
            }
        ],
        "project_id": "test_project",
        "chat_id": f"test_chat_{int(datetime.now().timestamp())}"
    }
    
    print(f"\n📤 Sending {len(payload['diff_chunks'])} DiffChunks to aggregator...")
    print(f"   Project ID: {payload['project_id']}")
    print(f"   Chat ID: {payload['chat_id']}")
    
    try:
        response = requests.post(
            f"{AGGREGATOR_URL}/triggers/memory/ingest_code_changes",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"\n📊 Result:")
            print(json.dumps(result, indent=2))
            
            # Verify data
            print(f"\n🔍 Verification:")
            print(f"   - Request UUID: {result.get('request_uuid', 'N/A')}")
            print(f"   - Code changes count: {result.get('code_changes_count', 0)}")
            print(f"   - Project ID: {result.get('project_id', 'N/A')}")
            
            print(f"\n📝 Next Steps:")
            print(f"   1. Check Neo4j Browser with queries:")
            print(f"      MATCH (cc:CodeChange {{project_id: 'test_project'}})")
            print(f"      RETURN cc.name, cc.file_path, cc.change_type, cc.severity")
            print(f"      ORDER BY cc.created_at DESC")
            print(f"      LIMIT 10")
            print(f"\n   2. Check Request node:")
            print(f"      MATCH (r:Request {{project_id: 'test_project'}})")
            print(f"      RETURN r.request_id, r.chat_id, r.created_at")
            print(f"      ORDER BY r.created_at DESC")
            print(f"      LIMIT 5")
            
        else:
            print(f"❌ FAILED!")
            print(f"\n📄 Response:")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Cannot connect to {AGGREGATOR_URL}")
        print(f"   Make sure MCP Aggregator is running on port 9003")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def check_aggregator_health():
    """Check if aggregator is running"""
    print("\n" + "=" * 70)
    print("HEALTH CHECK: MCP Aggregator")
    print("=" * 70)
    
    try:
        response = requests.get(f"{AGGREGATOR_URL}/admin/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Aggregator is healthy")
            print(f"   Servers: {health.get('servers', {})}")
            return True
        else:
            print(f"⚠️  Aggregator returned {response.status_code}")
            return False
    except:
        print(f"❌ Aggregator is not running on {AGGREGATOR_URL}")
        return False


def check_zepai_health():
    """Check if ZepAI Memory Server is running"""
    print("\n" + "=" * 70)
    print("HEALTH CHECK: ZepAI Memory Server")
    print("=" * 70)
    
    ZEPAI_URL = "http://localhost:9001"  # Port from config
    
    try:
        response = requests.get(f"{ZEPAI_URL}/", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ ZepAI Memory Server is healthy")
            print(f"   Service: {info.get('service', 'Unknown')}")
            print(f"   Version: {info.get('version', 'Unknown')}")
            return True
        else:
            print(f"⚠️  ZepAI returned {response.status_code}")
            return False
    except:
        print(f"❌ ZepAI Memory Server is not running on {ZEPAI_URL}")
        return False


if __name__ == "__main__":
    print("\n" + "🧪 " + "=" * 68)
    print("   TESTING CODE CHANGES INGESTION TO NEO4J")
    print("=" * 70 + "\n")
    
    # Check services health first
    aggregator_ok = check_aggregator_health()
    zepai_ok = check_zepai_health()
    
    if not aggregator_ok:
        print("\n❌ Aggregator is not running. Start it with:")
        print("   cd mcp-aggregator && python aggregator_server.py")
        exit(1)
    
    if not zepai_ok:
        print("\n❌ ZepAI Memory Server is not running. Start it with:")
        print("   cd ZepAI/fastmcp_server && python server_http.py")
        exit(1)
    
    # Run the test
    test_ingest_code_changes()
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70 + "\n")
