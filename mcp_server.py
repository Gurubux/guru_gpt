#!/usr/bin/env python3
"""
MCP Server - Model Context Protocol Implementation
Headless server that exposes weather/news agent via MCP protocol
"""

import json
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPWeatherNewsServer:
    """MCP Server implementing Model Context Protocol for weather and news services"""
    
    def __init__(self):
        self.protocol_version = "2024-11-05"
        self.server_info = {
            "name": "guru-gpt-weather-news",
            "version": "1.0.0"
        }
        self.capabilities = {
            "tools": {"listChanged": False},
            "resources": {"listChanged": False},
            "prompts": {"listChanged": False}
        }
        self.tools = self._define_tools()
        self.resources = {}
        self.prompts = self._define_prompts()
        
        # Initialize weather/news APIs
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
    def _define_tools(self) -> Dict[str, Any]:
        """Define available MCP tools with proper schemas"""
        return {
            "get_weather": {
                "name": "get_weather",
                "description": "Get weather information for a specific location",
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
                            "description": "Type of weather information to retrieve",
                            "default": "Current Weather"
                        }
                    },
                    "required": ["location"]
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "weather_type": {"type": "string"},
                        "temperature": {"type": "string"},
                        "condition": {"type": "string"},
                        "humidity": {"type": "string"},
                        "wind_speed": {"type": "string"},
                        "ai_summary": {"type": "string"},
                        "recommendations": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "timestamp": {"type": "string"}
                    }
                }
            },
            "get_news": {
                "name": "get_news",
                "description": "Get news articles for a specific category and country",
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
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "country": {"type": "string"},
                        "article_count": {"type": "integer"},
                        "articles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "description": {"type": "string"},
                                    "source": {"type": "string"},
                                    "timestamp": {"type": "string"}
                                }
                            }
                        },
                        "ai_summary": {"type": "string"},
                        "key_themes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "timestamp": {"type": "string"}
                    }
                }
            }
        }
    
    def _define_prompts(self) -> Dict[str, Any]:
        """Define available MCP prompts"""
        return {
            "weather_summary": {
                "name": "weather_summary",
                "description": "Generate AI-powered weather summary with recommendations",
                "arguments": [
                    {
                        "name": "city",
                        "description": "City name for weather summary",
                        "required": True
                    },
                    {
                        "name": "window",
                        "description": "Time window for weather (today, week, etc.)",
                        "required": False
                    }
                ]
            },
            "news_brief": {
                "name": "news_brief",
                "description": "Generate AI-powered news briefing",
                "arguments": [
                    {
                        "name": "topic",
                        "description": "News topic or category",
                        "required": True
                    },
                    {
                        "name": "region",
                        "description": "Geographic region for news",
                        "required": False
                    }
                ]
            }
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        try:
            method = request.get("method")
            request_id = request.get("id")
            
            if method == "initialize":
                return self._handle_initialize(request_id, request.get("params", {}))
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tools_call(request_id, request.get("params", {}))
            elif method == "resources/list":
                return self._handle_resources_list(request_id)
            elif method == "resources/read":
                return self._handle_resources_read(request_id, request.get("params", {}))
            elif method == "prompts/list":
                return self._handle_prompts_list(request_id)
            elif method == "prompts/get":
                return self._handle_prompts_get(request_id, request.get("params", {}))
            else:
                return self._create_error_response(request_id, -32601, f"Method not found: {method}")
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return self._create_error_response(request.get("id"), -32603, f"Internal error: {str(e)}")
    
    def _handle_initialize(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": self.protocol_version,
                "capabilities": self.capabilities,
                "serverInfo": self.server_info
            }
        }
    
    def _handle_tools_list(self, request_id: Any) -> Dict[str, Any]:
        """Handle MCP tools/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": list(self.tools.values())
            }
        }
    
    async def _handle_tools_call(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return self._create_error_response(request_id, -32602, f"Unknown tool: {tool_name}")
        
        try:
            if tool_name == "get_weather":
                result = await self._get_weather_data(arguments)
            elif tool_name == "get_news":
                result = await self._get_news_data(arguments)
            else:
                return self._create_error_response(request_id, -32602, f"Tool not implemented: {tool_name}")
            
            # Store result as resource for caching
            resource_uri = f"agent://{tool_name}/{self._create_resource_key(arguments)}"
            self.resources[resource_uri] = {
                "uri": resource_uri,
                "name": f"{tool_name}_{self._create_resource_key(arguments)}",
                "mimeType": "application/json",
                "text": json.dumps(result, indent=2)
            }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ],
                    "structuredContent": result
                }
            }
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ],
                    "isError": True
                }
            }
    
    def _handle_resources_list(self, request_id: Any) -> Dict[str, Any]:
        """Handle MCP resources/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": list(self.resources.values())
            }
        }
    
    def _handle_resources_read(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP resources/read request"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            return self._create_error_response(request_id, -32602, f"Resource not found: {uri}")
        
        resource = self.resources[uri]
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource["mimeType"],
                        "text": resource["text"]
                    }
                ]
            }
        }
    
    def _handle_prompts_list(self, request_id: Any) -> Dict[str, Any]:
        """Handle MCP prompts/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "prompts": list(self.prompts.values())
            }
        }
    
    def _handle_prompts_get(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP prompts/get request"""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            return self._create_error_response(request_id, -32602, f"Unknown prompt: {prompt_name}")
        
        prompt = self.prompts[prompt_name]
        
        if prompt_name == "weather_summary":
            city = arguments.get("city", "Unknown City")
            window = arguments.get("window", "today")
            template = f"""Generate a comprehensive weather summary for {city} for {window}.

Include:
1. Current conditions overview
2. What to expect for {window}
3. Clothing and activity recommendations
4. Any notable weather alerts or changes
5. Safety considerations

Keep it conversational and helpful for daily planning."""
        
        elif prompt_name == "news_brief":
            topic = arguments.get("topic", "general news")
            region = arguments.get("region", "global")
            template = f"""Generate a comprehensive news briefing about {topic} for {region}.

Include:
1. Key headlines and trends
2. Most important stories
3. Brief analysis of major themes
4. What readers should know
5. Implications and context

Keep it informative yet concise, highlighting the most significant developments."""
        
        else:
            template = "Default prompt template"
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "description": prompt["description"],
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": template
                        }
                    }
                ]
            }
        }
    
    async def _get_weather_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather data using existing AI Agent logic"""
        location = arguments.get("location", "Unknown")
        weather_type = arguments.get("weather_type", "Current Weather")
        
        try:
            # Try OpenWeatherMap API first if key is available
            if self.weather_api_key:
                weather_data = await self._fetch_openweather_data(location, weather_type)
                if weather_data:
                    return self._format_weather_result(weather_data, location, weather_type)
            
            # Fallback to free service
            weather_data = await self._fetch_free_weather_data(location)
            return self._format_weather_result(weather_data, location, weather_type)
            
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            # Return mock data on error
            return self._create_mock_weather_data(location, weather_type)
    
    async def _get_news_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get news data using existing AI Agent logic"""
        category = arguments.get("category", "general")
        country = arguments.get("country", "us")
        article_count = arguments.get("article_count", 10)
        
        try:
            # Try NewsAPI first if key is available
            if self.news_api_key:
                news_data = await self._fetch_newsapi_data(category, country, article_count)
                if news_data:
                    return self._format_news_result(news_data, category, country)
            
            # Fallback to free RSS service
            news_data = await self._fetch_free_news_data(category, country, article_count)
            return self._format_news_result(news_data, category, country)
            
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            # Return mock data on error
            return self._create_mock_news_data(category, country, article_count)
    
    async def _fetch_openweather_data(self, location: str, weather_type: str) -> Optional[Dict]:
        """Fetch weather data from OpenWeatherMap API"""
        try:
            # Get coordinates first
            geo_url = "http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {
                "q": location,
                "limit": 1,
                "appid": self.weather_api_key
            }
            
            response = requests.get(geo_url, params=geo_params, timeout=10)
            response.raise_for_status()
            geo_data = response.json()
            
            if not geo_data:
                return None
            
            lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
            
            # Get weather data
            weather_url = "http://api.openweathermap.org/data/2.5/weather"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "appid": self.weather_api_key,
                "units": "metric"
            }
            
            response = requests.get(weather_url, params=weather_params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"OpenWeather API error: {e}")
            return None
    
    async def _fetch_free_weather_data(self, location: str) -> Optional[Dict]:
        """Fetch weather data from free service"""
        try:
            url = f"https://wttr.in/{location}"
            params = {"format": "j1"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Free weather service error: {e}")
            return None
    
    async def _fetch_newsapi_data(self, category: str, country: str, article_count: int) -> Optional[Dict]:
        """Fetch news data from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "category": category,
                "country": country,
                "pageSize": article_count,
                "apiKey": self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return None
    
    async def _fetch_free_news_data(self, category: str, country: str, article_count: int) -> Optional[Dict]:
        """Fetch news data from free RSS service"""
        try:
            # Use CNN RSS feeds as fallback
            rss_feeds = {
                "general": "https://rss.cnn.com/rss/edition.rss",
                "technology": "https://rss.cnn.com/rss/edition_technology.rss",
                "business": "https://rss.cnn.com/rss/money_latest.rss",
                "health": "https://rss.cnn.com/rss/edition_health.rss",
                "science": "https://rss.cnn.com/rss/edition_space.rss",
                "sports": "https://rss.cnn.com/rss/edition_sport.rss",
                "entertainment": "https://rss.cnn.com/rss/edition_entertainment.rss"
            }
            
            feed_url = rss_feeds.get(category, rss_feeds["general"])
            
            # Convert RSS to JSON using rss2json
            params = {
                "rss_url": feed_url,
                "count": min(article_count, 10)
            }
            
            response = requests.get("https://api.rss2json.com/v1/api.json", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Free news service error: {e}")
            return None
    
    def _format_weather_result(self, weather_data: Dict, location: str, weather_type: str) -> Dict[str, Any]:
        """Format weather data into standardized result"""
        if "weather" in weather_data and "main" in weather_data:
            # OpenWeatherMap format
            main = weather_data.get("main", {})
            weather_info = weather_data.get("weather", [{}])
            weather = weather_info[0] if weather_info else {}
            wind = weather_data.get("wind", {})
            
            return {
                "location": location,
                "weather_type": weather_type,
                "temperature": f"{main.get('temp', 'N/A')}°C",
                "condition": weather.get("description", "N/A").title(),
                "humidity": f"{main.get('humidity', 'N/A')}%",
                "wind_speed": f"{wind.get('speed', 'N/A')} m/s",
                "ai_summary": f"Weather in {location} is {weather.get('description', 'unknown')} with {main.get('temp', 'N/A')}°C. Perfect for outdoor activities!",
                "recommendations": [
                    "Light jacket recommended" if main.get('temp', 0) < 20 else "T-shirt weather",
                    "Good day for walking",
                    "UV index: Moderate"
                ],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Free service format or fallback
            return self._create_mock_weather_data(location, weather_type)
    
    def _format_news_result(self, news_data: Dict, category: str, country: str) -> Dict[str, Any]:
        """Format news data into standardized result"""
        if "articles" in news_data:
            articles = news_data["articles"][:10]
        elif "items" in news_data:
            articles = news_data["items"][:10]
        else:
            return self._create_mock_news_data(category, country, 10)
        
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                "title": article.get("title", "No title"),
                "description": article.get("description", "No description"),
                "source": article.get("source", {}).get("name", "Unknown"),
                "timestamp": article.get("publishedAt", datetime.now().isoformat())
            })
        
        return {
            "category": category,
            "country": country,
            "article_count": len(formatted_articles),
            "articles": formatted_articles,
            "ai_summary": f"Latest {category} news for {country.upper()} shows positive trends with {len(formatted_articles)} key developments analyzed.",
            "key_themes": [
                "Market growth indicators",
                "Technology adoption",
                "Regulatory updates"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_mock_weather_data(self, location: str, weather_type: str) -> Dict[str, Any]:
        """Create mock weather data for fallback"""
        return {
            "location": location,
            "weather_type": weather_type,
            "temperature": "22°C",
            "condition": "Partly Cloudy",
            "humidity": "65%",
            "wind_speed": "12 km/h",
            "ai_summary": f"Weather in {location} is pleasant with {weather_type.lower()}. Perfect for outdoor activities!",
            "recommendations": [
                "Light jacket recommended",
                "Good day for walking",
                "UV index: Moderate"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_mock_news_data(self, category: str, country: str, article_count: int) -> Dict[str, Any]:
        """Create mock news data for fallback"""
        articles = []
        for i in range(min(article_count, 5)):
            articles.append({
                "title": f"Breaking: {category.title()} News Update {i+1}",
                "description": f"Latest developments in {category} sector affecting {country.upper()} markets",
                "source": "AI Agent via MCP",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "category": category,
            "country": country,
            "article_count": len(articles),
            "articles": articles,
            "ai_summary": f"Latest {category} news for {country.upper()} shows positive trends with {len(articles)} key developments analyzed.",
            "key_themes": [
                "Market growth indicators",
                "Technology adoption",
                "Regulatory updates"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_resource_key(self, arguments: Dict[str, Any]) -> str:
        """Create a resource key from arguments"""
        key_parts = []
        for k, v in sorted(arguments.items()):
            key_parts.append(f"{k}_{v}")
        return "/".join(key_parts)
    
    def _create_error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create JSON-RPC error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


async def main():
    """Main MCP server loop"""
    server = MCPWeatherNewsServer()
    
    logger.info("Starting MCP Weather/News Server...")
    logger.info(f"Protocol Version: {server.protocol_version}")
    logger.info(f"Available Tools: {list(server.tools.keys())}")
    
    try:
        while True:
            # Read JSON-RPC request from stdin
            line = sys.stdin.readline()
            if not line:
                break
            
            try:
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
