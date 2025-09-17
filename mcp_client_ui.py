#!/usr/bin/env python3
"""
MCP Client UI - Streamlit interface for testing MCP server
"""

import streamlit as st
import json
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

class MCPClientUI:
    """Streamlit UI for MCP client testing"""
    
    def __init__(self):
        self.server_process = None
        self.request_id = 0
    
    def start_server(self):
        """Start the MCP server process"""
        try:
            self.server_process = subprocess.Popen(
                ["python", "mcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(1)  # Give server time to start
            return True
        except Exception as e:
            st.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP server process"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
    
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
    
    def render_ui(self):
        """Render the MCP client UI"""
        st.title("🔌 MCP Client - Weather/News Server")
        st.markdown("*Test the Model Context Protocol server implementation*")
        
        # Server Control
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start MCP Server", disabled=self.server_process is not None):
                if self.start_server():
                    st.success("MCP Server started!")
                    st.rerun()
                else:
                    st.error("Failed to start server")
        
        with col2:
            if st.button("⏹️ Stop MCP Server", disabled=self.server_process is None):
                self.stop_server()
                st.success("MCP Server stopped!")
                st.rerun()
        
        with col3:
            if st.button("🔄 Restart Server"):
                self.stop_server()
                time.sleep(0.5)
                if self.start_server():
                    st.success("MCP Server restarted!")
                    st.rerun()
        
        # Server Status
        st.markdown("---")
        st.subheader("📊 Server Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            status = "🟢 Running" if self.server_process else "🔴 Stopped"
            st.metric("Server Status", status)
        
        with status_col2:
            st.metric("Request ID", self.request_id)
        
        if not self.server_process:
            st.warning("⚠️ Start the MCP Server to test functionality")
            return
        
        # Initialize Connection
        if st.button("🔗 Initialize Connection"):
            with st.spinner("Initializing MCP connection..."):
                response = self.send_request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "streamlit-client",
                        "version": "1.0.0"
                    }
                })
                
                if "result" in response:
                    st.success("✅ MCP connection initialized!")
                    st.json(response["result"])
                else:
                    st.error(f"❌ Initialization failed: {response}")
        
        # Tools Testing
        st.markdown("---")
        st.subheader("🛠️ Tools Testing")
        
        # List Tools
        if st.button("📋 List Available Tools"):
            with st.spinner("Fetching tools..."):
                response = self.send_request("tools/list")
                
                if "result" in response:
                    tools = response["result"].get("tools", [])
                    st.success(f"✅ Found {len(tools)} tools")
                    
                    for tool in tools:
                        with st.expander(f"🔧 {tool['name']}", expanded=False):
                            st.write(f"**Description:** {tool['description']}")
                            st.write("**Input Schema:**")
                            st.json(tool.get("inputSchema", {}))
                            if "outputSchema" in tool:
                                st.write("**Output Schema:**")
                                st.json(tool.get("outputSchema", {}))
                else:
                    st.error(f"❌ Tools list failed: {response}")
        
        # Weather Tool
        st.markdown("#### 🌤️ Weather Tool")
        col1, col2 = st.columns(2)
        
        with col1:
            weather_location = st.text_input("Location:", value="London, UK", key="weather_loc")
        with col2:
            weather_type = st.selectbox(
                "Weather Type:",
                ["Current Weather", "5-Day Forecast", "Weather Summary"],
                key="weather_type"
            )
        
        if st.button("🌤️ Get Weather"):
            with st.spinner("Calling weather tool..."):
                response = self.send_request("tools/call", {
                    "name": "get_weather",
                    "arguments": {
                        "location": weather_location,
                        "weather_type": weather_type
                    }
                })
                
                if "result" in response:
                    st.success("✅ Weather tool call successful!")
                    
                    # Show raw response
                    with st.expander("📄 Raw MCP Response", expanded=False):
                        st.json(response)
                    
                    # Show structured content
                    structured = response["result"].get("structuredContent", {})
                    if structured:
                        st.subheader("🌤️ Weather Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Temperature", structured.get("temperature", "N/A"))
                        with col2:
                            st.metric("Condition", structured.get("condition", "N/A"))
                        with col3:
                            st.metric("Humidity", structured.get("humidity", "N/A"))
                        
                        st.write("**AI Summary:**", structured.get("ai_summary", "N/A"))
                        
                        recommendations = structured.get("recommendations", [])
                        if recommendations:
                            st.write("**Recommendations:**")
                            for rec in recommendations:
                                st.write(f"• {rec}")
                else:
                    st.error(f"❌ Weather tool failed: {response}")
        
        # News Tool
        st.markdown("#### 📰 News Tool")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            news_category = st.selectbox(
                "Category:",
                ["general", "business", "technology", "health", "science", "sports", "entertainment"],
                key="news_category"
            )
        with col2:
            news_country = st.selectbox(
                "Country:",
                ["us", "gb", "ca", "au", "fr", "de", "in", "jp"],
                key="news_country"
            )
        with col3:
            news_count = st.slider("Article Count:", 5, 20, 10, key="news_count")
        
        if st.button("📰 Get News"):
            with st.spinner("Calling news tool..."):
                response = self.send_request("tools/call", {
                    "name": "get_news",
                    "arguments": {
                        "category": news_category,
                        "country": news_country,
                        "article_count": news_count
                    }
                })
                
                if "result" in response:
                    st.success("✅ News tool call successful!")
                    
                    # Show raw response
                    with st.expander("📄 Raw MCP Response", expanded=False):
                        st.json(response)
                    
                    # Show structured content
                    structured = response["result"].get("structuredContent", {})
                    if structured:
                        st.subheader("📰 News Results")
                        st.write(f"**Category:** {structured.get('category', 'N/A').title()}")
                        st.write(f"**Country:** {structured.get('country', 'N/A').upper()}")
                        st.write(f"**AI Summary:** {structured.get('ai_summary', 'N/A')}")
                        
                        key_themes = structured.get("key_themes", [])
                        if key_themes:
                            st.write("**Key Themes:**")
                            for theme in key_themes:
                                st.write(f"• {theme}")
                        
                        articles = structured.get("articles", [])
                        if articles:
                            st.write("**Articles:**")
                            for i, article in enumerate(articles, 1):
                                st.write(f"**{i}. {article.get('title', 'No title')}**")
                                st.write(f"*{article.get('description', 'No description')}*")
                                st.write(f"Source: {article.get('source', 'Unknown')} | {article.get('timestamp', 'Unknown time')}")
                                st.markdown("---")
                else:
                    st.error(f"❌ News tool failed: {response}")
        
        # Resources Testing
        st.markdown("---")
        st.subheader("📚 Resources Testing")
        
        if st.button("📋 List Resources"):
            with st.spinner("Fetching resources..."):
                response = self.send_request("resources/list")
                
                if "result" in response:
                    resources = response["result"].get("resources", [])
                    st.success(f"✅ Found {len(resources)} cached resources")
                    
                    for resource in resources:
                        with st.expander(f"📄 {resource['name']}", expanded=False):
                            st.write(f"**URI:** {resource['uri']}")
                            st.write(f"**MIME Type:** {resource['mimeType']}")
                            
                            if st.button(f"Read {resource['name']}", key=f"read_{resource['uri']}"):
                                read_response = self.send_request("resources/read", {"uri": resource["uri"]})
                                if "result" in read_response:
                                    contents = read_response["result"].get("contents", [])
                                    if contents:
                                        st.json(json.loads(contents[0]["text"]))
                                else:
                                    st.error(f"Failed to read resource: {read_response}")
                else:
                    st.error(f"❌ Resources list failed: {response}")
        
        # Prompts Testing
        st.markdown("---")
        st.subheader("📝 Prompts Testing")
        
        if st.button("📋 List Prompts"):
            with st.spinner("Fetching prompts..."):
                response = self.send_request("prompts/list")
                
                if "result" in response:
                    prompts = response["result"].get("prompts", [])
                    st.success(f"✅ Found {len(prompts)} prompts")
                    
                    for prompt in prompts:
                        with st.expander(f"📝 {prompt['name']}", expanded=False):
                            st.write(f"**Description:** {prompt['description']}")
                            
                            # Show arguments
                            args = prompt.get("arguments", [])
                            if args:
                                st.write("**Arguments:**")
                                for arg in args:
                                    required = " (required)" if arg.get("required", False) else ""
                                    st.write(f"• {arg['name']}: {arg['description']}{required}")
                            
                            # Test prompt
                            if prompt["name"] == "weather_summary":
                                city = st.text_input("City:", value="Paris", key=f"prompt_city_{prompt['name']}")
                                window = st.text_input("Window:", value="today", key=f"prompt_window_{prompt['name']}")
                                
                                if st.button(f"Get {prompt['name']}", key=f"get_prompt_{prompt['name']}"):
                                    prompt_response = self.send_request("prompts/get", {
                                        "name": prompt["name"],
                                        "arguments": {
                                            "city": city,
                                            "window": window
                                        }
                                    })
                                    
                                    if "result" in prompt_response:
                                        st.success("✅ Prompt retrieved!")
                                        messages = prompt_response["result"].get("messages", [])
                                        if messages:
                                            content = messages[0].get("content", {})
                                            st.text_area("Prompt Template:", content.get("text", ""), height=200)
                                    else:
                                        st.error(f"Failed to get prompt: {prompt_response}")
                            
                            elif prompt["name"] == "news_brief":
                                topic = st.text_input("Topic:", value="technology", key=f"prompt_topic_{prompt['name']}")
                                region = st.text_input("Region:", value="global", key=f"prompt_region_{prompt['name']}")
                                
                                if st.button(f"Get {prompt['name']}", key=f"get_prompt_{prompt['name']}"):
                                    prompt_response = self.send_request("prompts/get", {
                                        "name": prompt["name"],
                                        "arguments": {
                                            "topic": topic,
                                            "region": region
                                        }
                                    })
                                    
                                    if "result" in prompt_response:
                                        st.success("✅ Prompt retrieved!")
                                        messages = prompt_response["result"].get("messages", [])
                                        if messages:
                                            content = messages[0].get("content", {})
                                            st.text_area("Prompt Template:", content.get("text", ""), height=200)
                                    else:
                                        st.error(f"Failed to get prompt: {prompt_response}")
                else:
                    st.error(f"❌ Prompts list failed: {response}")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="MCP Client UI",
        page_icon="🔌",
        layout="wide"
    )
    
    # Initialize client
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClientUI()
    
    # Render UI
    st.session_state.mcp_client.render_ui()


if __name__ == "__main__":
    main()
