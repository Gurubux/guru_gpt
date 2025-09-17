#!/usr/bin/env python3
"""
MCP Client - Test client for the MCP Weather/News Server
"""

import json
import subprocess
import sys
from typing import Dict, Any, List

class MCPClient:
    """Simple MCP client for testing the weather/news server"""
    
    def __init__(self, server_command: List[str]):
        """Initialize MCP client with server command"""
        self.server_command = server_command
        self.request_id = 0
        self.server_process = None
    
    def start_server(self):
        """Start the MCP server process"""
        try:
            self.server_process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✅ MCP Server started")
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
        return True
    
    def stop_server(self):
        """Stop the MCP server process"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("✅ MCP Server stopped")
    
    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a request to the MCP server"""
        if not self.server_process:
            return {"error": "Server not started"}
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        
        if params:
            request["params"] = params
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            # Read response
            response_line = self.server_process.stdout.readline()
            if not response_line:
                return {"error": "No response from server"}
            
            response = json.loads(response_line.strip())
            return response
            
        except Exception as e:
            return {"error": f"Request failed: {e}"}
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP connection"""
        return self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
    
    def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        return self.send_request("tools/list")
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool"""
        return self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    def list_resources(self) -> Dict[str, Any]:
        """List available resources"""
        return self.send_request("resources/list")
    
    def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        return self.send_request("resources/read", {"uri": uri})
    
    def list_prompts(self) -> Dict[str, Any]:
        """List available prompts"""
        return self.send_request("prompts/list")
    
    def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a prompt"""
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self.send_request("prompts/get", params)


def test_mcp_server():
    """Test the MCP server functionality"""
    print("🧪 Testing MCP Weather/News Server")
    print("=" * 50)
    
    # Start server
    client = MCPClient(["python", "mcp_server.py"])
    if not client.start_server():
        return
    
    try:
        # Test 1: Initialize
        print("\n1️⃣ Testing Initialize...")
        init_response = client.initialize()
        if "result" in init_response:
            print("✅ Initialize successful")
            print(f"   Protocol: {init_response['result'].get('protocolVersion')}")
            print(f"   Server: {init_response['result'].get('serverInfo', {}).get('name')}")
        else:
            print(f"❌ Initialize failed: {init_response}")
            return
        
        # Test 2: List Tools
        print("\n2️⃣ Testing Tools List...")
        tools_response = client.list_tools()
        if "result" in tools_response:
            tools = tools_response["result"].get("tools", [])
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print(f"❌ Tools list failed: {tools_response}")
        
        # Test 3: Call Weather Tool
        print("\n3️⃣ Testing Weather Tool...")
        weather_response = client.call_tool("get_weather", {
            "location": "Paris, France",
            "weather_type": "Current Weather"
        })
        if "result" in weather_response:
            print("✅ Weather tool successful")
            result = weather_response["result"].get("structuredContent", {})
            print(f"   Location: {result.get('location')}")
            print(f"   Temperature: {result.get('temperature')}")
            print(f"   Condition: {result.get('condition')}")
        else:
            print(f"❌ Weather tool failed: {weather_response}")
        
        # Test 4: Call News Tool
        print("\n4️⃣ Testing News Tool...")
        news_response = client.call_tool("get_news", {
            "category": "technology",
            "country": "us",
            "article_count": 5
        })
        if "result" in news_response:
            print("✅ News tool successful")
            result = news_response["result"].get("structuredContent", {})
            print(f"   Category: {result.get('category')}")
            print(f"   Country: {result.get('country')}")
            print(f"   Articles: {result.get('article_count')}")
        else:
            print(f"❌ News tool failed: {news_response}")
        
        # Test 5: List Resources
        print("\n5️⃣ Testing Resources List...")
        resources_response = client.list_resources()
        if "result" in resources_response:
            resources = resources_response["result"].get("resources", [])
            print(f"✅ Found {len(resources)} cached resources")
            for resource in resources:
                print(f"   - {resource['name']}")
        else:
            print(f"❌ Resources list failed: {resources_response}")
        
        # Test 6: List Prompts
        print("\n6️⃣ Testing Prompts List...")
        prompts_response = client.list_prompts()
        if "result" in prompts_response:
            prompts = prompts_response["result"].get("prompts", [])
            print(f"✅ Found {len(prompts)} prompts:")
            for prompt in prompts:
                print(f"   - {prompt['name']}: {prompt['description']}")
        else:
            print(f"❌ Prompts list failed: {prompts_response}")
        
        # Test 7: Get Prompt
        print("\n7️⃣ Testing Prompt Get...")
        prompt_response = client.get_prompt("weather_summary", {
            "city": "London",
            "window": "today"
        })
        if "result" in prompt_response:
            print("✅ Prompt get successful")
            messages = prompt_response["result"].get("messages", [])
            if messages:
                content = messages[0].get("content", {})
                text = content.get("text", "")
                print(f"   Template preview: {text[:100]}...")
        else:
            print(f"❌ Prompt get failed: {prompt_response}")
        
        print("\n🎉 MCP Server testing completed!")
        
    finally:
        client.stop_server()


if __name__ == "__main__":
    test_mcp_server()
