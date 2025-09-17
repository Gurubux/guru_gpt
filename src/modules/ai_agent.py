"""
AI Agent Module
Fetches live weather or news via free APIs and summarizes it with AI
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os


class AIAgent:
    """AI Agent that can fetch weather and news data and provide AI-powered summaries"""
    
    def __init__(self):
        """Initialize the AI Agent"""
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Free API endpoints
        self.weather_base_url = "http://api.openweathermap.org/data/2.5"
        self.news_base_url = "https://newsapi.org/v2"
        
        # Alternative free APIs (no key required)
        self.free_weather_url = "https://wttr.in"  # Weather in terminal format
        self.free_news_url = "https://api.rss2json.com/v1/api.json"  # RSS to JSON
    
    def render_agent_interface(self):
        """Render the AI agent interface"""
        st.header("🤖 AI Agent - Weather & News Assistant")
        st.markdown("*Get live weather and news updates with AI-powered summaries*")
        
        # Service selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌤️ Weather Service")
            self.render_weather_interface()
        
        with col2:
            st.subheader("📰 News Service")
            self.render_news_interface()
        
        st.markdown("---")
        
        # Recent queries section
        self.render_recent_queries()
    
    def render_weather_interface(self):
        """Render weather fetching interface"""
        # Location input
        location = st.text_input(
            "Enter location (city, country):",
            value="London, UK",
            key="weather_location",
            help="Enter city name and country (e.g., 'Paris, France')"
        )
        
        # Weather type selection
        weather_type = st.selectbox(
            "Weather Information:",
            ["Current Weather", "5-Day Forecast", "Weather Summary"],
            key="weather_type"
        )
        
        # Debug mode toggle
        debug_mode = st.checkbox("🔍 Debug Mode", key="weather_debug", help="Show detailed API responses for troubleshooting")
        
        # Fetch weather button
        if st.button("🌤️ Get Weather", key="fetch_weather"):
            if location.strip():
                with st.spinner("Fetching weather data..."):
                    weather_data = self.fetch_weather_data(location, weather_type)
                    if weather_data:
                        if debug_mode:
                            st.info("✅ Raw API Response:")
                            st.json(weather_data)
                        self.display_weather_results(weather_data, location, weather_type)
                    else:
                        st.error("Failed to fetch weather data. Please check the location and try again.")
            else:
                st.warning("Please enter a location")
    
    def render_news_interface(self):
        """Render news fetching interface"""
        # News category selection
        category = st.selectbox(
            "News Category:",
            ["general", "business", "technology", "health", "science", "sports", "entertainment"],
            key="news_category"
        )
        
        # Country selection for localized news
        country = st.selectbox(
            "Country:",
            ["us", "gb", "ca", "au", "fr", "de", "in", "jp"],
            format_func=lambda x: {
                "us": "United States", "gb": "United Kingdom", "ca": "Canada",
                "au": "Australia", "fr": "France", "de": "Germany", 
                "in": "India", "jp": "Japan"
            }.get(x, x),
            key="news_country"
        )
        
        # Number of articles
        article_count = st.slider(
            "Number of articles:",
            min_value=5,
            max_value=20,
            value=10,
            key="article_count"
        )
        
        # Debug mode toggle
        debug_mode_news = st.checkbox("🔍 Debug Mode", key="news_debug", help="Show detailed API responses for troubleshooting")
        
        # Fetch news button
        if st.button("📰 Get News", key="fetch_news"):
            with st.spinner("Fetching latest news..."):
                news_data = self.fetch_news_data(category, country, article_count)
                if news_data:
                    if debug_mode_news:
                        st.info("✅ Raw API Response:")
                        st.json(news_data)
                    self.display_news_results(news_data, category, country)
                else:
                    st.error("Failed to fetch news data. Please try again.")
    
    def fetch_weather_data(self, location: str, weather_type: str) -> Optional[Dict]:
        """Fetch weather data from APIs"""
        try:
            # Try OpenWeatherMap API first if key is available
            if self.weather_api_key:
                result = self._fetch_openweather_data(location, weather_type)
                if result:
                    return result
                else:
                    st.warning("OpenWeatherMap API failed, trying free service...")
                    return self._fetch_free_weather_data(location)
            else:
                # Use free weather service
                return self._fetch_free_weather_data(location)
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")
            return None
    
    def _fetch_openweather_data(self, location: str, weather_type: str) -> Optional[Dict]:
        """Fetch weather data from OpenWeatherMap API"""
        try:
            # Get coordinates first
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {
                "q": location,
                "limit": 1,
                "appid": self.weather_api_key
            }
            
            geo_response = requests.get(geo_url, params=geo_params, timeout=10)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data:
                st.error(f"Location '{location}' not found. Please check the spelling and try again.")
                return None
            
            lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
            
            if weather_type == "Current Weather":
                url = f"{self.weather_base_url}/weather"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.weather_api_key,
                    "units": "metric"
                }
            elif weather_type == "5-Day Forecast":
                url = f"{self.weather_base_url}/forecast"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.weather_api_key,
                    "units": "metric"
                }
            else:  # Weather Summary
                url = f"{self.weather_base_url}/weather"
                params = {
                    "lat": lat,
                    "lon": lon,
                    "appid": self.weather_api_key,
                    "units": "metric"
                }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather_result = response.json()
            
            # Check if the response has the expected structure
            if "cod" in weather_result and weather_result["cod"] != 200:
                st.error(f"OpenWeather API error: {weather_result.get('message', 'Unknown error')}")
                return None
                
            return weather_result
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error with OpenWeather API: {str(e)}")
            return None
        except KeyError as e:
            st.error(f"Unexpected response format from OpenWeather API: missing {str(e)}")
            return None
        except Exception as e:
            st.error(f"OpenWeather API error: {str(e)}")
            return None
    
    def _fetch_free_weather_data(self, location: str) -> Optional[Dict]:
        """Fetch weather data from free service (wttr.in)"""
        try:
            # wttr.in provides weather in JSON format
            url = f"{self.free_weather_url}/{location}"
            params = {"format": "j1"}  # JSON format
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            # Check if we got valid data
            if not weather_data or "current_condition" not in weather_data:
                st.error(f"No weather data found for '{location}'. Please check the location name.")
                return None
                
            return weather_data
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error with free weather service: {str(e)}")
            return None
        except ValueError as e:
            st.error(f"Invalid response from weather service: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Free weather service error: {str(e)}")
            return None
    
    def fetch_news_data(self, category: str, country: str, article_count: int) -> Optional[Dict]:
        """Fetch news data from APIs"""
        try:
            # Try NewsAPI first if key is available
            if self.news_api_key:
                return self._fetch_newsapi_data(category, country, article_count)
            else:
                # Use free RSS service
                return self._fetch_free_news_data(category, country, article_count)
        except Exception as e:
            st.error(f"Error fetching news data: {str(e)}")
            return None
    
    def _fetch_newsapi_data(self, category: str, country: str, article_count: int) -> Optional[Dict]:
        """Fetch news data from NewsAPI"""
        try:
            url = f"{self.news_base_url}/top-headlines"
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
            st.error(f"NewsAPI error: {str(e)}")
            return None
    
    def _fetch_free_news_data(self, category: str, country: str, article_count: int) -> Optional[Dict]:
        """Fetch news data from free RSS service"""
        try:
            # Updated RSS feed URLs with reliable sources
            rss_feeds = {
                "general": {
                    "us": "https://rss.cnn.com/rss/edition.rss",
                    "gb": "http://feeds.bbci.co.uk/news/rss.xml",
                    "ca": "https://www.cbc.ca/cmlink/rss-topstories",
                    "au": "https://www.abc.net.au/news/feed/1534/rss.xml",
                    "default": "https://rss.cnn.com/rss/edition.rss"
                },
                "technology": {
                    "default": "https://rss.cnn.com/rss/edition_technology.rss"
                },
                "business": {
                    "default": "https://rss.cnn.com/rss/money_latest.rss"
                },
                "health": {
                    "default": "https://rss.cnn.com/rss/edition_health.rss"
                },
                "science": {
                    "default": "https://rss.cnn.com/rss/edition_space.rss"
                },
                "sports": {
                    "default": "https://rss.cnn.com/rss/edition_sport.rss"
                },
                "entertainment": {
                    "default": "https://rss.cnn.com/rss/edition_entertainment.rss"
                }
            }
            
            # Get appropriate RSS feed
            feed_url = rss_feeds.get(category, {}).get(country)
            if not feed_url:
                feed_url = rss_feeds.get(category, {}).get("default")
                if not feed_url:
                    feed_url = rss_feeds["general"]["default"]
            
            # Try rss2json.com first with corrected parameters
            try:
                params = {
                    "rss_url": feed_url,
                    "count": min(article_count, 10)  # Limit to 10 for free service
                }
                
                response = requests.get(self.free_news_url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok" and data.get("items"):
                        return data
                
            except Exception as e:
                st.warning(f"RSS-to-JSON service issue: {str(e)}")
            
            # Fallback: Try to parse RSS directly using a simple parser
            st.info("Trying alternative RSS parsing method...")
            return self._parse_rss_directly(feed_url, article_count)
            
        except Exception as e:
            st.error(f"Free news service error: {str(e)}")
            return None
    
    def _parse_rss_directly(self, feed_url: str, article_count: int) -> Optional[Dict]:
        """Parse RSS feed directly as a fallback method"""
        try:
            import xml.etree.ElementTree as ET
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(feed_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Find all item elements
            items = []
            for item in root.findall('.//item')[:article_count]:
                title_elem = item.find('title')
                desc_elem = item.find('description')
                link_elem = item.find('link')
                pubdate_elem = item.find('pubDate')
                
                article = {
                    "title": title_elem.text if title_elem is not None else "No title",
                    "description": desc_elem.text if desc_elem is not None else "No description",
                    "link": link_elem.text if link_elem is not None else "#",
                    "pubDate": pubdate_elem.text if pubdate_elem is not None else "Unknown date"
                }
                items.append(article)
            
            return {
                "status": "ok",
                "items": items,
                "source": "direct_rss_parsing"
            }
            
        except Exception as e:
            st.error(f"Direct RSS parsing failed: {str(e)}")
            # Return a mock news structure for demo purposes
            return self._get_mock_news_data(article_count)
    
    def _get_mock_news_data(self, article_count: int) -> Dict:
        """Return mock news data when all other methods fail"""
        from datetime import datetime
        
        mock_articles = [
            {
                "title": "Breaking: AI Technology Continues to Advance",
                "description": "Artificial Intelligence technologies are making significant strides across various industries, transforming how we work and live.",
                "link": "#",
                "pubDate": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            },
            {
                "title": "Global Markets Show Mixed Results",
                "description": "Financial markets around the world are displaying varied performance as investors react to economic indicators.",
                "link": "#", 
                "pubDate": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            },
            {
                "title": "Climate Change Research Reveals New Insights",
                "description": "Scientists have published new findings about climate patterns and their impact on global weather systems.",
                "link": "#",
                "pubDate": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            },
            {
                "title": "Healthcare Innovation Improves Patient Outcomes",
                "description": "New medical technologies and treatment approaches are helping healthcare providers deliver better patient care.",
                "link": "#",
                "pubDate": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            },
            {
                "title": "Education Technology Transforms Learning",
                "description": "Digital learning platforms and educational tools are revolutionizing how students access and engage with information.",
                "link": "#",
                "pubDate": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            }
        ]
        
        return {
            "status": "demo",
            "items": mock_articles[:article_count],
            "source": "mock_data",
            "message": "Using demo news data - RSS feeds temporarily unavailable"
        }
    
    def display_weather_results(self, weather_data: Dict, location: str, weather_type: str):
        """Display weather results with AI summary"""
        st.success(f"✅ Weather data fetched for {location}")
        
        # Display raw weather information
        with st.expander("🌤️ Weather Details", expanded=True):
            try:
                if "weather" in weather_data and "main" in weather_data:  # OpenWeatherMap format
                    self._display_openweather_results(weather_data, weather_type)
                elif "current_condition" in weather_data:  # wttr.in format
                    self._display_free_weather_results(weather_data)
                else:
                    # Handle unexpected format - show raw data
                    st.warning("⚠️ Unexpected weather data format received")
                    st.json(weather_data)
            except Exception as e:
                st.error(f"Error displaying weather data: {str(e)}")
                st.json(weather_data)  # Show raw data for debugging
        
        # Generate AI summary
        self._generate_weather_summary(weather_data, location, weather_type)
        
        # Store in recent queries
        self._store_recent_query("weather", {
            "location": location,
            "type": weather_type,
            "timestamp": datetime.now(),
            "data": weather_data
        })
    
    def _display_openweather_results(self, weather_data: Dict, weather_type: str):
        """Display OpenWeatherMap results"""
        try:
            if weather_type == "Current Weather":
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    main_data = weather_data.get("main", {})
                    temp = main_data.get("temp", "N/A")
                    feels_like = main_data.get("feels_like", "N/A")
                    if temp != "N/A" and feels_like != "N/A":
                        st.metric("Temperature", f"{temp}°C", f"Feels like {feels_like}°C")
                    else:
                        st.metric("Temperature", f"{temp}°C")
                
                with col2:
                    humidity = main_data.get("humidity", "N/A")
                    pressure = main_data.get("pressure", "N/A")
                    st.metric("Humidity", f"{humidity}%" if humidity != "N/A" else "N/A")
                    st.metric("Pressure", f"{pressure} hPa" if pressure != "N/A" else "N/A")
                
                with col3:
                    weather_info = weather_data.get("weather", [{}])
                    description = weather_info[0].get("description", "N/A").title() if weather_info else "N/A"
                    wind_data = weather_data.get("wind", {})
                    wind_speed = wind_data.get("speed", "N/A")
                    st.metric("Condition", description)
                    st.metric("Wind Speed", f"{wind_speed} m/s" if wind_speed != "N/A" else "N/A")
            
            elif weather_type == "5-Day Forecast":
                st.subheader("📅 5-Day Forecast")
                forecast_list = weather_data.get("list", [])
                if forecast_list:
                    for i, forecast in enumerate(forecast_list[:5]):
                        dt = forecast.get("dt")
                        if dt:
                            date = datetime.fromtimestamp(dt).strftime("%Y-%m-%d %H:%M")
                        else:
                            date = f"Day {i+1}"
                        
                        main_data = forecast.get("main", {})
                        temp = main_data.get("temp", "N/A")
                        weather_info = forecast.get("weather", [{}])
                        description = weather_info[0].get("description", "N/A").title() if weather_info else "N/A"
                        st.write(f"**{date}**: {temp}°C - {description}")
                else:
                    st.warning("No forecast data available")
            
        except Exception as e:
            st.error(f"Error displaying OpenWeather data: {str(e)}")
            st.json(weather_data)
    
    def _display_free_weather_results(self, weather_data: Dict):
        """Display free weather service results"""
        try:
            current_conditions = weather_data.get("current_condition", [])
            if not current_conditions:
                st.warning("No current weather data available")
                return
                
            current = current_conditions[0]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temp = current.get("temp_C", "N/A")
                feels_like = current.get("FeelsLikeC", "N/A")
                if temp != "N/A" and feels_like != "N/A":
                    st.metric("Temperature", f"{temp}°C", f"Feels like {feels_like}°C")
                else:
                    st.metric("Temperature", f"{temp}°C" if temp != "N/A" else "N/A")
            
            with col2:
                humidity = current.get("humidity", "N/A")
                pressure = current.get("pressure", "N/A")
                st.metric("Humidity", f"{humidity}%" if humidity != "N/A" else "N/A")
                st.metric("Pressure", f"{pressure} mb" if pressure != "N/A" else "N/A")
            
            with col3:
                weather_desc = current.get("weatherDesc", [])
                description = weather_desc[0].get("value", "N/A") if weather_desc else "N/A"
                wind_speed = current.get("windspeedKmph", "N/A")
                st.metric("Condition", description)
                st.metric("Wind Speed", f"{wind_speed} km/h" if wind_speed != "N/A" else "N/A")
                
        except Exception as e:
            st.error(f"Error displaying free weather data: {str(e)}")
            st.json(weather_data)
    
    def display_news_results(self, news_data: Dict, category: str, country: str):
        """Display news results with AI summary"""
        # Show different messages based on data source
        source = news_data.get("source", "unknown")
        if source == "mock_data":
            st.info(f"📰 Demo news for {category} category (RSS feeds temporarily unavailable)")
            if "message" in news_data:
                st.caption(news_data["message"])
        elif source == "direct_rss_parsing":
            st.success(f"✅ Latest {category} news fetched via direct RSS parsing")
        else:
            st.success(f"✅ Latest {category} news fetched")
        
        # Display news articles
        with st.expander("📰 News Articles", expanded=True):
            if "articles" in news_data:  # NewsAPI format
                articles = news_data["articles"]
            elif "items" in news_data:  # RSS format or parsed format
                articles = news_data["items"]
            else:
                articles = []
            
            if not articles:
                st.warning("No articles found for this category and country combination.")
                return
            
            for i, article in enumerate(articles[:10], 1):
                title = article.get("title", "No title")
                description = article.get("description") or article.get("content", "No description")
                url = article.get("url") or article.get("link", "#")
                published = article.get("publishedAt") or article.get("pubDate", "Unknown date")
                
                # Clean up description text (remove HTML tags if present)
                if description and len(description) > 300:
                    description = description[:300] + "..."
                
                st.markdown(f"**{i}. {title}**")
                if description != "No description":
                    st.markdown(f"*{description}*")
                
                if url != "#":
                    st.markdown(f"[Read more]({url}) | Published: {published}")
                else:
                    st.markdown(f"Published: {published}")
                st.markdown("---")
        
        # Generate AI summary
        self._generate_news_summary(news_data, category, country)
        
        # Store in recent queries
        self._store_recent_query("news", {
            "category": category,
            "country": country,
            "timestamp": datetime.now(),
            "data": news_data
        })
    
    def _generate_weather_summary(self, weather_data: Dict, location: str, weather_type: str):
        """Generate AI-powered weather summary"""
        if not st.session_state.get("chatbot"):
            st.warning("AI summarization not available - OpenAI API key not configured")
            return
        
        with st.expander("🤖 AI Weather Summary", expanded=True):
            with st.spinner("Generating AI summary..."):
                # Prepare weather data for AI
                weather_text = self._format_weather_for_ai(weather_data, location, weather_type)
                
                # Create prompt for AI summarization
                prompt = f"""
                Please provide a concise, friendly weather summary based on this data:

                {weather_text}

                Include:
                1. Current conditions overview
                2. What to expect today
                3. Clothing/activity recommendations
                4. Any notable weather alerts or changes

                Keep it conversational and helpful for daily planning.
                """
                
                try:
                    chatbot = st.session_state.chatbot
                    response, _ = chatbot.get_response(
                        [{"role": "user", "content": prompt}],
                        model=st.session_state.get("current_model", "gpt-3.5-turbo"),
                        temperature=0.7
                    )
                    
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Failed to generate AI summary: {str(e)}")
    
    def _generate_news_summary(self, news_data: Dict, category: str, country: str):
        """Generate AI-powered news summary"""
        if not st.session_state.get("chatbot"):
            st.warning("AI summarization not available - OpenAI API key not configured")
            return
        
        with st.expander("🤖 AI News Summary", expanded=True):
            with st.spinner("Generating AI summary..."):
                # Prepare news data for AI
                news_text = self._format_news_for_ai(news_data, category, country)
                
                # Create prompt for AI summarization
                prompt = f"""
                Please provide a comprehensive news summary based on these articles:

                {news_text}

                Include:
                1. Key headlines and trends
                2. Most important stories
                3. Brief analysis of major themes
                4. What readers should know

                Keep it informative yet concise, highlighting the most significant developments.
                """
                
                try:
                    chatbot = st.session_state.chatbot
                    response, _ = chatbot.get_response(
                        [{"role": "user", "content": prompt}],
                        model=st.session_state.get("current_model", "gpt-3.5-turbo"),
                        temperature=0.7
                    )
                    
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Failed to generate AI summary: {str(e)}")
    
    def _format_weather_for_ai(self, weather_data: Dict, location: str, weather_type: str) -> str:
        """Format weather data for AI processing"""
        formatted_text = f"Weather for {location} ({weather_type}):\n\n"
        
        try:
            if "weather" in weather_data and "main" in weather_data:  # OpenWeatherMap format
                main = weather_data.get("main", {})
                weather_info = weather_data.get("weather", [{}])
                weather = weather_info[0] if weather_info else {}
                wind = weather_data.get("wind", {})
                
                temp = main.get("temp", "N/A")
                feels_like = main.get("feels_like", "N/A")
                condition = weather.get("description", "N/A")
                humidity = main.get("humidity", "N/A")
                pressure = main.get("pressure", "N/A")
                wind_speed = wind.get("speed", "N/A")
                
                formatted_text += f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
                formatted_text += f"Condition: {condition}\n"
                formatted_text += f"Humidity: {humidity}%\n"
                formatted_text += f"Pressure: {pressure} hPa\n"
                formatted_text += f"Wind Speed: {wind_speed} m/s\n"
                
            elif "current_condition" in weather_data:  # wttr.in format
                current_conditions = weather_data.get("current_condition", [])
                if current_conditions:
                    current = current_conditions[0]
                    temp = current.get("temp_C", "N/A")
                    feels_like = current.get("FeelsLikeC", "N/A")
                    weather_desc = current.get("weatherDesc", [{}])
                    condition = weather_desc[0].get("value", "N/A") if weather_desc else "N/A"
                    humidity = current.get("humidity", "N/A")
                    pressure = current.get("pressure", "N/A")
                    wind_speed = current.get("windspeedKmph", "N/A")
                    
                    formatted_text += f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
                    formatted_text += f"Condition: {condition}\n"
                    formatted_text += f"Humidity: {humidity}%\n"
                    formatted_text += f"Pressure: {pressure} mb\n"
                    formatted_text += f"Wind Speed: {wind_speed} km/h\n"
                else:
                    formatted_text += "No current weather data available\n"
            else:
                formatted_text += "Weather data format not recognized\n"
                
        except Exception as e:
            formatted_text += f"Error processing weather data: {str(e)}\n"
        
        return formatted_text
    
    def _format_news_for_ai(self, news_data: Dict, category: str, country: str) -> str:
        """Format news data for AI processing"""
        formatted_text = f"Latest {category} news for {country}:\n\n"
        
        if "articles" in news_data:  # NewsAPI format
            articles = news_data["articles"][:5]  # Limit to first 5 for AI processing
        elif "items" in news_data:  # RSS format
            articles = news_data["items"][:5]
        else:
            return formatted_text + "No articles found."
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            description = article.get("description") or article.get("content", "No description")
            
            formatted_text += f"{i}. {title}\n"
            formatted_text += f"   {description[:300]}...\n\n" if len(description) > 300 else f"   {description}\n\n"
        
        return formatted_text
    
    def _store_recent_query(self, query_type: str, data: Dict):
        """Store recent query for history"""
        if "agent_history" not in st.session_state:
            st.session_state.agent_history = []
        
        query_record = {
            "type": query_type,
            "timestamp": datetime.now(),
            "data": data
        }
        
        st.session_state.agent_history.insert(0, query_record)
        
        # Keep only last 10 queries
        if len(st.session_state.agent_history) > 10:
            st.session_state.agent_history = st.session_state.agent_history[:10]
    
    def render_recent_queries(self):
        """Render recent queries section"""
        if "agent_history" not in st.session_state or not st.session_state.agent_history:
            return
        
        st.subheader("📊 Recent Queries")
        
        with st.expander("View Recent Queries", expanded=False):
            for i, query in enumerate(st.session_state.agent_history[:5], 1):
                query_type = query["type"]
                timestamp = query["timestamp"].strftime("%Y-%m-%d %H:%M")
                data = query["data"]
                
                if query_type == "weather":
                    st.markdown(f"**{i}. 🌤️ Weather for {data['location']}** - {timestamp}")
                    st.markdown(f"   Type: {data['type']}")
                elif query_type == "news":
                    st.markdown(f"**{i}. 📰 News ({data['category']}, {data['country']})** - {timestamp}")
                
                if st.button(f"🔄 Repeat Query", key=f"repeat_{i}"):
                    if query_type == "weather":
                        st.session_state.weather_location = data['location']
                        st.session_state.weather_type = data['type']
                    elif query_type == "news":
                        st.session_state.news_category = data['category']
                        st.session_state.news_country = data['country']
                    st.rerun()
                
                st.markdown("---")
