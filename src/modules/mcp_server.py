"""
MCP (Model Context Protocol) Server Implementation
Demonstrates the evolution from AI Agents to MCP Servers
"""

import streamlit as st
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time


class MCPServer:
    """
    Simple MCP Server that wraps our AI Agent functionality
    Demonstrates how AI Agents can be exposed as MCP servers
    """
    
    def __init__(self):
        """Initialize the MCP Server"""
        self.server_id = f"guru-gpt-agent-{int(time.time())}"
        self.capabilities = {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for (e.g., 'Paris, France')"
                            },
                            "weather_type": {
                                "type": "string",
                                "enum": ["Current Weather", "5-Day Forecast", "Weather Summary"],
                                "description": "Type of weather information to retrieve"
                            }
                        },
                        "required": ["location"]
                    }
                },
                {
                    "name": "get_news",
                    "description": "Get news articles for a category and country",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["general", "business", "technology", "health", "science", "sports", "entertainment"],
                                "description": "News category to retrieve"
                            },
                            "country": {
                                "type": "string",
                                "enum": ["us", "gb", "ca", "au", "fr", "de", "in", "jp"],
                                "description": "Country for localized news"
                            },
                            "article_count": {
                                "type": "integer",
                                "minimum": 5,
                                "maximum": 20,
                                "default": 10,
                                "description": "Number of articles to retrieve"
                            }
                        },
                        "required": ["category", "country"]
                    }
                },
                {
                    "name": "get_agent_status",
                    "description": "Get current status of the AI Agent",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ],
            "resources": [
                {
                    "uri": "agent://weather",
                    "name": "Weather Agent",
                    "description": "Access to weather intelligence capabilities",
                    "mimeType": "application/json"
                },
                {
                    "uri": "agent://news",
                    "name": "News Agent", 
                    "description": "Access to news intelligence capabilities",
                    "mimeType": "application/json"
                }
            ]
        }
        self.request_history = []
        self.is_running = False
    
    def start_server(self):
        """Start the MCP server"""
        self.is_running = True
        self.request_history.append({
            "timestamp": datetime.now(),
            "action": "server_started",
            "message": f"MCP Server {self.server_id} started successfully"
        })
    
    def stop_server(self):
        """Stop the MCP server"""
        self.is_running = False
        self.request_history.append({
            "timestamp": datetime.now(),
            "action": "server_stopped",
            "message": f"MCP Server {self.server_id} stopped"
        })
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "server_id": self.server_id,
            "status": "running" if self.is_running else "stopped",
            "capabilities": self.capabilities,
            "uptime": len(self.request_history),
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        if not self.is_running:
            return {
                "error": "Server not running",
                "success": False
            }
        
        # Log the request
        self.request_history.append({
            "timestamp": datetime.now(),
            "action": "tool_call",
            "tool": tool_name,
            "parameters": parameters
        })
        
        try:
            if tool_name == "get_weather":
                return self._handle_weather_request(parameters)
            elif tool_name == "get_news":
                return self._handle_news_request(parameters)
            elif tool_name == "get_agent_status":
                return self._handle_status_request(parameters)
            else:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "success": False
                }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _handle_weather_request(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weather tool requests"""
        location = parameters.get("location", "Unknown")
        weather_type = parameters.get("weather_type", "Current Weather")
        
        # Simulate weather data (in real implementation, this would call the AI Agent)
        weather_data = {
            "location": location,
            "type": weather_type,
            "temperature": "22°C",
            "condition": "Partly Cloudy",
            "humidity": "65%",
            "wind_speed": "12 km/h",
            "ai_summary": f"Weather in {location} is pleasant with {weather_type.lower()}. Perfect for outdoor activities!",
            "recommendations": [
                "Light jacket recommended",
                "Good day for walking",
                "UV index: Moderate"
            ]
        }
        
        return {
            "success": True,
            "data": weather_data,
            "mcp_response": {
                "tool": "get_weather",
                "result": weather_data,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _handle_news_request(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle news tool requests"""
        category = parameters.get("category", "general")
        country = parameters.get("country", "us")
        article_count = parameters.get("article_count", 10)
        
        # Simulate news data (in real implementation, this would call the AI Agent)
        news_data = {
            "category": category,
            "country": country,
            "article_count": article_count,
            "articles": [
                {
                    "title": f"Breaking: {category.title()} News Update",
                    "description": f"Latest developments in {category} sector affecting {country.upper()} markets",
                    "source": "AI Agent via MCP",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "title": f"Market Analysis: {category.title()} Trends",
                    "description": f"Comprehensive analysis of {category} industry trends and future outlook",
                    "source": "AI Agent via MCP",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "ai_summary": f"Latest {category} news for {country.upper()} shows positive trends with {article_count} key developments analyzed.",
            "key_themes": [
                "Market growth indicators",
                "Technology adoption",
                "Regulatory updates"
            ]
        }
        
        return {
            "success": True,
            "data": news_data,
            "mcp_response": {
                "tool": "get_news",
                "result": news_data,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _handle_status_request(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status tool requests"""
        return {
            "success": True,
            "data": self.get_server_info(),
            "mcp_response": {
                "tool": "get_agent_status",
                "result": self.get_server_info(),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get request history"""
        return self.request_history
    
    def clear_history(self):
        """Clear request history"""
        self.request_history = []


class MCPClient:
    """
    Simple MCP Client to demonstrate how other applications can interact with our MCP Server
    """
    
    def __init__(self, server: MCPServer):
        """Initialize MCP Client with reference to server"""
        self.server = server
        self.client_id = f"client-{int(time.time())}"
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        return self.server.handle_tool_call(tool_name, parameters)
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server"""
        return self.server.capabilities.get("tools", [])
    
    def get_server_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities"""
        return self.server.capabilities


def render_mcp_interface():
    """Render the MCP interface tab"""
    st.header("🔌 MCP Server - Model Context Protocol")
    st.markdown("*Demonstrating the evolution from AI Agents to MCP Servers*")
    
    # Initialize MCP server in session state
    if "mcp_server" not in st.session_state:
        st.session_state.mcp_server = MCPServer()
    
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClient(st.session_state.mcp_server)
    
    server = st.session_state.mcp_server
    client = st.session_state.mcp_client
    
    # Server Control Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start MCP Server", disabled=server.is_running):
            server.start_server()
            st.success("MCP Server started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop MCP Server", disabled=not server.is_running):
            server.stop_server()
            st.success("MCP Server stopped!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Restart Server"):
            server.stop_server()
            time.sleep(0.5)
            server.start_server()
            st.success("MCP Server restarted!")
            st.rerun()
    
    # Server Status
    st.markdown("---")
    st.subheader("📊 Server Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        status = "🟢 Running" if server.is_running else "🔴 Stopped"
        st.metric("Server Status", status)
    
    with status_col2:
        st.metric("Server ID", server.server_id)
    
    with status_col3:
        st.metric("Total Requests", len(server.request_history))
    
    # MCP Tools Section
    st.markdown("---")
    st.subheader("🛠️ MCP Tools")
    
    if not server.is_running:
        st.warning("⚠️ Start the MCP Server to use tools")
    else:
        # Tool Selection
        available_tools = client.list_available_tools()
        tool_names = [tool["name"] for tool in available_tools]
        
        selected_tool = st.selectbox(
            "Select MCP Tool:",
            tool_names,
            help="Choose a tool to call on the MCP server"
        )
        
        # Display tool details
        if selected_tool:
            tool_info = next(tool for tool in available_tools if tool["name"] == selected_tool)
            
            with st.expander(f"📋 Tool Details: {selected_tool}", expanded=True):
                st.write(f"**Description:** {tool_info['description']}")
                st.write("**Parameters:**")
                st.json(tool_info["inputSchema"])
        
        # Tool Parameters and Execution
        st.markdown("#### 🎯 Execute Tool")
        
        if selected_tool == "get_weather":
            col1, col2 = st.columns(2)
            with col1:
                location = st.text_input("Location:", value="London, UK", key="mcp_weather_location")
            with col2:
                weather_type = st.selectbox(
                    "Weather Type:",
                    ["Current Weather", "5-Day Forecast", "Weather Summary"],
                    key="mcp_weather_type"
                )
            
            if st.button("🌤️ Get Weather via MCP"):
                with st.spinner("Calling MCP weather tool..."):
                    result = client.call_tool("get_weather", {
                        "location": location,
                        "weather_type": weather_type
                    })
                    
                    if result["success"]:
                        st.success("✅ MCP Tool call successful!")
                        st.json(result["mcp_response"])
                        
                        # Display formatted results
                        data = result["data"]
                        st.subheader("🌤️ Weather Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Temperature", data["temperature"])
                        with col2:
                            st.metric("Condition", data["condition"])
                        with col3:
                            st.metric("Humidity", data["humidity"])
                        
                        st.write("**AI Summary:**", data["ai_summary"])
                        st.write("**Recommendations:**")
                        for rec in data["recommendations"]:
                            st.write(f"• {rec}")
                    else:
                        st.error(f"❌ MCP Tool call failed: {result['error']}")
        
        elif selected_tool == "get_news":
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox(
                    "Category:",
                    ["general", "business", "technology", "health", "science", "sports", "entertainment"],
                    key="mcp_news_category"
                )
            with col2:
                country = st.selectbox(
                    "Country:",
                    ["us", "gb", "ca", "au", "fr", "de", "in", "jp"],
                    key="mcp_news_country"
                )
            with col3:
                article_count = st.slider("Article Count:", 5, 20, 10, key="mcp_news_count")
            
            if st.button("📰 Get News via MCP"):
                with st.spinner("Calling MCP news tool..."):
                    result = client.call_tool("get_news", {
                        "category": category,
                        "country": country,
                        "article_count": article_count
                    })
                    
                    if result["success"]:
                        st.success("✅ MCP Tool call successful!")
                        st.json(result["mcp_response"])
                        
                        # Display formatted results
                        data = result["data"]
                        st.subheader("📰 News Results")
                        st.write(f"**Category:** {data['category'].title()}")
                        st.write(f"**Country:** {data['country'].upper()}")
                        st.write(f"**AI Summary:** {data['ai_summary']}")
                        
                        st.write("**Key Themes:**")
                        for theme in data["key_themes"]:
                            st.write(f"• {theme}")
                        
                        st.write("**Articles:**")
                        for i, article in enumerate(data["articles"], 1):
                            st.write(f"**{i}. {article['title']}**")
                            st.write(f"*{article['description']}*")
                            st.write(f"Source: {article['source']} | {article['timestamp']}")
                            st.markdown("---")
                    else:
                        st.error(f"❌ MCP Tool call failed: {result['error']}")
        
        elif selected_tool == "get_agent_status":
            if st.button("📊 Get Agent Status via MCP"):
                with st.spinner("Calling MCP status tool..."):
                    result = client.call_tool("get_agent_status", {})
                    
                    if result["success"]:
                        st.success("✅ MCP Tool call successful!")
                        st.json(result["mcp_response"])
                        
                        # Display formatted status
                        data = result["data"]
                        st.subheader("🤖 Agent Status")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Server ID", data["server_id"])
                            st.metric("Status", data["status"])
                        with col2:
                            st.metric("Uptime", f"{data['uptime']} requests")
                            st.metric("Timestamp", data["timestamp"])
                    else:
                        st.error(f"❌ MCP Tool call failed: {result['error']}")
    
    # Request History
    st.markdown("---")
    st.subheader("📜 Request History")
    
    if server.request_history:
        history_df = []
        for req in server.request_history[-10:]:  # Show last 10 requests
            history_df.append({
                "Timestamp": req["timestamp"].strftime("%H:%M:%S"),
                "Action": req["action"],
                "Tool": req.get("tool", "N/A"),
                "Status": "✅" if req.get("success", True) else "❌"
            })
        
        if history_df:
            st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            server.clear_history()
            st.rerun()
    else:
        st.info("No requests yet. Start the server and call some tools!")
    
    # MCP Protocol Information
    st.markdown("---")
    st.subheader("📚 About MCP (Model Context Protocol)")
    
    with st.expander("What is MCP?", expanded=True):
        st.markdown("""
        **Model Context Protocol (MCP)** is a standardized way for AI models to interact with external tools and data sources.
        
        ### Key Concepts:
        - **Standardized Interface**: Common protocol for AI-tool communication
        - **Tool Discovery**: AI can discover available capabilities
        - **Structured Communication**: Well-defined request/response formats
        - **Extensibility**: Easy to add new tools and capabilities
        
        ### Benefits:
        - **Interoperability**: Works across different AI models and platforms
        - **Modularity**: Tools can be developed and deployed independently
        - **Scalability**: Easy to add new capabilities without changing core systems
        - **Reliability**: Standardized error handling and response formats
        """)
    
    with st.expander("MCP vs Traditional APIs", expanded=False):
        st.markdown("""
        | **Traditional APIs** | **MCP Servers** |
        |---------------------|-----------------|
        | Fixed endpoints | Dynamic tool discovery |
        | Manual integration | Automatic capability detection |
        | Custom protocols | Standardized communication |
        | Single-purpose | Multi-tool capabilities |
        | Human-driven | AI-driven interaction |
        """)
    
    with st.expander("Evolution Timeline", expanded=False):
        st.markdown("""
        ### 🕰️ AI Architecture Evolution:
        
        1. **Traditional APIs** (2000s)
           - REST/GraphQL endpoints
           - Manual integration
           - Human-driven usage
        
        2. **AI Agents** (2020s)
           - Autonomous decision making
           - Multi-source data fusion
           - Intelligent error recovery
        
        3. **MCP Servers** (2024+)
           - Standardized AI-tool communication
           - Dynamic capability discovery
           - AI-driven tool orchestration
        
        **Next Evolution**: Autonomous MCP networks with self-organizing capabilities
        """)
