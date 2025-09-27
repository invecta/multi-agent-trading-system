"""
Enhanced Sentiment Agent with NLP Models and Vector Database Integration
Implements sentiment analysis using Hugging Face transformers and Pinecone vector database
"""
import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import asyncio
import re
from collections import defaultdict
import hashlib

# NLP and ML libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: pinecone library not available. Install with: pip install pinecone-client")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not available. Install with: pip install openai")

from multi_agent_framework import BaseAgent, AgentOutput, TradingSignal, SignalType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Sentiment data structure"""
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    symbol: str
    metadata: Dict[str, Any]

@dataclass
class NewsArticle:
    """News article structure"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    symbol: str
    category: str

@dataclass
class SocialMediaPost:
    """Social media post structure"""
    text: str
    author: str
    platform: str
    timestamp: datetime
    symbol: str
    engagement: Dict[str, int]  # likes, retweets, etc.

class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "cpu"
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback sentiment analysis")
            return
        
        try:
            # Financial sentiment model
            self.models['financial'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=self.device
            )
            
            # General sentiment model
            self.models['general'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            
            # Emotion analysis model
            self.models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device
            )
            
            logger.info("Sentiment analysis models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    async def analyze_sentiment(self, text: str, model_type: str = 'financial') -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            if not text or len(text.strip()) < 10:
                return {
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.5,
                    'emotions': {}
                }
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            if model_type in self.models:
                # Use transformer model
                result = self.models[model_type](cleaned_text)
                
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]
                    label = prediction['label'].lower()
                    confidence = prediction['score']
                    
                    # Convert to sentiment score
                    if 'positive' in label or 'bullish' in label:
                        sentiment_score = confidence
                        sentiment_label = 'bullish'
                    elif 'negative' in label or 'bearish' in label:
                        sentiment_score = -confidence
                        sentiment_label = 'bearish'
                    else:
                        sentiment_score = 0.0
                        sentiment_label = 'neutral'
                else:
                    sentiment_score = 0.0
                    sentiment_label = 'neutral'
                    confidence = 0.5
            else:
                # Fallback sentiment analysis
                sentiment_score, sentiment_label, confidence = self._fallback_sentiment_analysis(cleaned_text)
            
            # Analyze emotions if available
            emotions = {}
            if 'emotion' in self.models:
                try:
                    emotion_result = self.models['emotion'](cleaned_text)
                    if isinstance(emotion_result, list) and len(emotion_result) > 0:
                        emotions = {emotion_result[0]['label']: emotion_result[0]['score']}
                except Exception as e:
                    logger.error(f"Error analyzing emotions: {e}")
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'emotions': emotions,
                'model_used': model_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.5,
                'emotions': {},
                'error': str(e)
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words
        words = text.split()
        words = [word for word in words if len(word) > 2]
        
        return ' '.join(words).strip()
    
    def _fallback_sentiment_analysis(self, text: str) -> Tuple[float, str, float]:
        """Fallback sentiment analysis using keyword matching"""
        bullish_keywords = [
            'bullish', 'buy', 'purchase', 'positive', 'growth', 'profit', 'gain',
            'rise', 'increase', 'up', 'strong', 'excellent', 'outperform', 'beat',
            'surge', 'rally', 'breakout', 'momentum', 'optimistic'
        ]
        
        bearish_keywords = [
            'bearish', 'sell', 'negative', 'decline', 'loss', 'fall', 'decrease',
            'down', 'weak', 'poor', 'underperform', 'miss', 'drop', 'crash',
            'correction', 'volatility', 'risk', 'concern', 'pessimistic'
        ]
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        
        total_keywords = bullish_count + bearish_count
        
        if total_keywords == 0:
            return 0.0, 'neutral', 0.5
        
        bullish_ratio = bullish_count / total_keywords
        bearish_ratio = bearish_count / total_keywords
        
        if bullish_ratio > bearish_ratio:
            sentiment_score = bullish_ratio
            sentiment_label = 'bullish'
            confidence = bullish_ratio
        elif bearish_ratio > bullish_ratio:
            sentiment_score = -bearish_ratio
            sentiment_label = 'bearish'
            confidence = bearish_ratio
        else:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
            confidence = 0.5
        
        return sentiment_score, sentiment_label, confidence

class VectorDatabase:
    """Vector database for storing and retrieving sentiment embeddings"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.pc = None
        self.index = None
        self.embedding_model = None
        
        if PINECONE_AVAILABLE:
            self._initialize_pinecone()
        else:
            logger.warning("Pinecone not available, using local storage")
            self.local_embeddings = {}
    
    def _initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            api_key = self.config.get('pinecone_api_key', 'demo-key')
            self.pc = Pinecone(api_key=api_key)
            
            # Create or get index
            index_name = self.config.get('pinecone_index_name', 'sentiment-embeddings')
            
            if index_name not in [index.name for index in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=index_name,
                    dimension=384,  # sentence-transformers dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            self.index = self.pc.Index(index_name)
            logger.info("Pinecone vector database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
    
    async def store_sentiment_embedding(self, sentiment_data: SentimentData, embedding: List[float]):
        """Store sentiment embedding in vector database"""
        try:
            # Create unique ID
            text_hash = hashlib.md5(sentiment_data.text.encode()).hexdigest()
            vector_id = f"{sentiment_data.symbol}_{text_hash}"
            
            metadata = {
                'symbol': sentiment_data.symbol,
                'sentiment_score': sentiment_data.sentiment_score,
                'sentiment_label': sentiment_data.sentiment_label,
                'confidence': sentiment_data.confidence,
                'source': sentiment_data.source,
                'timestamp': sentiment_data.timestamp.isoformat(),
                'text': sentiment_data.text[:500]  # Truncate for storage
            }
            
            if self.index:
                # Store in Pinecone
                self.index.upsert(
                    vectors=[(vector_id, embedding, metadata)]
                )
            else:
                # Store locally
                self.local_embeddings[vector_id] = {
                    'embedding': embedding,
                    'metadata': metadata
                }
            
            logger.debug(f"Stored sentiment embedding for {sentiment_data.symbol}")
            
        except Exception as e:
            logger.error(f"Error storing sentiment embedding: {e}")
    
    async def search_similar_sentiments(self, query_embedding: List[float], symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar sentiments"""
        try:
            if self.index:
                # Search in Pinecone
                results = self.index.query(
                    vector=query_embedding,
                    filter={'symbol': symbol},
                    top_k=limit,
                    include_metadata=True
                )
                
                return [
                    {
                        'id': match['id'],
                        'score': match['score'],
                        'metadata': match['metadata']
                    }
                    for match in results['matches']
                ]
            else:
                # Search locally
                symbol_embeddings = {
                    k: v for k, v in self.local_embeddings.items()
                    if v['metadata']['symbol'] == symbol
                }
                
                # Calculate cosine similarity
                similarities = []
                for vector_id, data in symbol_embeddings.items():
                    similarity = np.dot(query_embedding, data['embedding']) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(data['embedding'])
                    )
                    similarities.append({
                        'id': vector_id,
                        'score': float(similarity),
                        'metadata': data['metadata']
                    })
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x['score'], reverse=True)
                return similarities[:limit]
                
        except Exception as e:
            logger.error(f"Error searching similar sentiments: {e}")
            return []

class NewsCollector:
    """Collects news articles from various sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_keys = {
            'newsapi': self.config.get('newsapi_key', 'demo-key'),
            'alpha_vantage': self.config.get('alpha_vantage_key', 'demo-key')
        }
    
    async def collect_news(self, symbol: str, hours_back: int = 24) -> List[NewsArticle]:
        """Collect news articles for a symbol"""
        articles = []
        
        # Collect from multiple sources
        tasks = [
            self._collect_from_newsapi(symbol, hours_back),
            self._collect_from_alpha_vantage(symbol, hours_back),
            self._collect_from_rss_feeds(symbol, hours_back)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error collecting news: {result}")
        
        # Remove duplicates and sort by timestamp
        unique_articles = self._deduplicate_articles(articles)
        return sorted(unique_articles, key=lambda x: x.published_at, reverse=True)
    
    async def _collect_from_newsapi(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Collect news from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'apiKey': self.api_keys['newsapi'],
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        
                        for article in data.get('articles', []):
                            articles.append(NewsArticle(
                                title=article.get('title', ''),
                                content=article.get('description', ''),
                                url=article.get('url', ''),
                                source=article.get('source', {}).get('name', 'NewsAPI'),
                                published_at=datetime.fromisoformat(
                                    article.get('publishedAt', datetime.now().isoformat()).replace('Z', '+00:00')
                                ),
                                symbol=symbol,
                                category='news'
                            ))
                        
                        return articles
                    else:
                        logger.error(f"NewsAPI error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error collecting from NewsAPI: {e}")
            return []
    
    async def _collect_from_alpha_vantage(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Collect news from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_keys['alpha_vantage'],
                'limit': 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        
                        for article in data.get('feed', []):
                            articles.append(NewsArticle(
                                title=article.get('title', ''),
                                content=article.get('summary', ''),
                                url=article.get('url', ''),
                                source=article.get('source', 'Alpha Vantage'),
                                published_at=datetime.fromisoformat(
                                    article.get('time_published', datetime.now().isoformat())
                                ),
                                symbol=symbol,
                                category='news'
                            ))
                        
                        return articles
                    else:
                        logger.error(f"Alpha Vantage error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error collecting from Alpha Vantage: {e}")
            return []
    
    async def _collect_from_rss_feeds(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Collect news from RSS feeds"""
        # Simplified RSS collection - in production, would use feedparser
        return []
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles

class SocialMediaCollector:
    """Collects social media posts and sentiment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # In production, would integrate with Twitter API, Reddit API, etc.
    
    async def collect_social_posts(self, symbol: str, hours_back: int = 24) -> List[SocialMediaPost]:
        """Collect social media posts for a symbol"""
        # Simplified implementation - in production would use actual APIs
        posts = []
        
        # Simulate social media posts
        sample_posts = [
            f"$AAPL looking bullish today! Strong earnings report expected.",
            f"Market volatility concerns for {symbol}. Risk management is key.",
            f"Technical analysis shows {symbol} breaking resistance levels.",
            f"Fundamental analysis suggests {symbol} is undervalued.",
            f"Market sentiment turning negative for {symbol} sector."
        ]
        
        for i, text in enumerate(sample_posts):
            posts.append(SocialMediaPost(
                text=text,
                author=f"user_{i}",
                platform="twitter",
                timestamp=datetime.now() - timedelta(hours=i),
                symbol=symbol,
                engagement={'likes': np.random.randint(0, 100), 'retweets': np.random.randint(0, 50)}
            ))
        
        return posts

class EnhancedSentimentAgent(BaseAgent):
    """Enhanced Sentiment Agent with NLP and vector database"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("enhanced_sentiment_agent", config)
        self.add_dependency("market_data_agent")
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.vector_db = VectorDatabase(config)
        self.news_collector = NewsCollector(config)
        self.social_collector = SocialMediaCollector(config)
        
        # Sentiment tracking
        self.sentiment_history = defaultdict(list)
        self.sentiment_trends = {}
        
    async def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Process sentiment analysis"""
        start_time = datetime.now()
        self.status = "running"
        
        try:
            market_data = input_data.get('market_data', {})
            all_signals = []
            sentiment_results = {}
            
            for symbol, data in market_data.items():
                logger.info(f"Analyzing sentiment for {symbol}")
                
                # Collect sentiment data
                sentiment_data = await self._collect_sentiment_data(symbol)
                
                # Analyze sentiment
                analysis_results = await self._analyze_sentiment_data(sentiment_data, symbol)
                
                # Generate trading signals
                signals = await self._generate_sentiment_signals(analysis_results, symbol)
                all_signals.extend(signals)
                
                # Store results
                sentiment_results[symbol] = {
                    'sentiment_data': sentiment_data,
                    'analysis_results': analysis_results,
                    'signals': signals
                }
                
                # Update sentiment trends
                self._update_sentiment_trends(symbol, analysis_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            output = AgentOutput(
                agent_id=self.agent_id,
                status="completed",
                data={
                    'sentiment_results': sentiment_results,
                    'sentiment_trends': self.sentiment_trends,
                    'total_signals': len(all_signals)
                },
                signals=all_signals,
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.last_output = output
            self.execution_history.append(output)
            return output
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Sentiment Agent error: {e}")
            return AgentOutput(
                agent_id=self.agent_id,
                status="error",
                data={},
                signals=[],
                risk_metrics=None,
                timestamp=datetime.now(),
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )
    
    async def _collect_sentiment_data(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment data from various sources"""
        sentiment_data = []
        
        # Collect news articles
        news_articles = await self.news_collector.collect_news(symbol, hours_back=24)
        
        # Collect social media posts
        social_posts = await self.social_collector.collect_social_posts(symbol, hours_back=24)
        
        # Analyze news sentiment
        for article in news_articles:
            analysis = await self.sentiment_analyzer.analyze_sentiment(
                f"{article.title} {article.content}", 'financial'
            )
            
            sentiment_data.append(SentimentData(
                text=f"{article.title} {article.content}",
                sentiment_score=analysis['sentiment_score'],
                sentiment_label=analysis['sentiment_label'],
                confidence=analysis['confidence'],
                source=article.source,
                timestamp=article.published_at,
                symbol=symbol,
                metadata={
                    'type': 'news',
                    'url': article.url,
                    'emotions': analysis.get('emotions', {})
                }
            ))
        
        # Analyze social media sentiment
        for post in social_posts:
            analysis = await self.sentiment_analyzer.analyze_sentiment(post.text, 'general')
            
            sentiment_data.append(SentimentData(
                text=post.text,
                sentiment_score=analysis['sentiment_score'],
                sentiment_label=analysis['sentiment_label'],
                confidence=analysis['confidence'],
                source=post.platform,
                timestamp=post.timestamp,
                symbol=symbol,
                metadata={
                    'type': 'social',
                    'author': post.author,
                    'engagement': post.engagement,
                    'emotions': analysis.get('emotions', {})
                }
            ))
        
        return sentiment_data
    
    async def _analyze_sentiment_data(self, sentiment_data: List[SentimentData], symbol: str) -> Dict[str, Any]:
        """Analyze collected sentiment data"""
        if not sentiment_data:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.5,
                'sources': {},
                'trend': 'stable'
            }
        
        # Calculate weighted average sentiment
        total_weight = 0
        weighted_sentiment = 0
        
        source_sentiments = defaultdict(list)
        
        for data in sentiment_data:
            # Weight by confidence and recency
            recency_weight = max(0.1, 1.0 - (datetime.now() - data.timestamp).total_seconds() / (24 * 3600))
            weight = data.confidence * recency_weight
            
            weighted_sentiment += data.sentiment_score * weight
            total_weight += weight
            
            source_sentiments[data.source].append(data.sentiment_score)
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Determine sentiment label
        if overall_sentiment > 0.2:
            sentiment_label = 'bullish'
        elif overall_sentiment < -0.2:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence
        confidence = min(0.9, total_weight / len(sentiment_data))
        
        # Analyze trend
        trend = self._analyze_sentiment_trend(symbol, overall_sentiment)
        
        # Calculate source-specific sentiments
        source_analysis = {}
        for source, sentiments in source_sentiments.items():
            source_analysis[source] = {
                'average_sentiment': np.mean(sentiments),
                'count': len(sentiments),
                'volatility': np.std(sentiments)
            }
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'sources': source_analysis,
            'trend': trend,
            'total_sources': len(sentiment_data),
            'data_points': len(sentiment_data)
        }
    
    def _analyze_sentiment_trend(self, symbol: str, current_sentiment: float) -> str:
        """Analyze sentiment trend"""
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []
        
        self.sentiment_history[symbol].append(current_sentiment)
        
        # Keep only recent history
        if len(self.sentiment_history[symbol]) > 10:
            self.sentiment_history[symbol] = self.sentiment_history[symbol][-10:]
        
        if len(self.sentiment_history[symbol]) < 3:
            return 'stable'
        
        # Calculate trend
        recent_sentiments = self.sentiment_history[symbol][-3:]
        trend_slope = np.polyfit(range(len(recent_sentiments)), recent_sentiments, 1)[0]
        
        if trend_slope > 0.1:
            return 'improving'
        elif trend_slope < -0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    async def _generate_sentiment_signals(self, analysis_results: Dict[str, Any], symbol: str) -> List[TradingSignal]:
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        overall_sentiment = analysis_results['overall_sentiment']
        sentiment_label = analysis_results['sentiment_label']
        confidence = analysis_results['confidence']
        trend = analysis_results['trend']
        
        # Generate signals based on sentiment strength and trend
        if sentiment_label == 'bullish' and confidence > 0.6:
            if trend == 'improving':
                signal_type = SignalType.STRONG_BUY
                signal_confidence = min(0.9, confidence + 0.1)
            else:
                signal_type = SignalType.BUY
                signal_confidence = confidence
        elif sentiment_label == 'bearish' and confidence > 0.6:
            if trend == 'deteriorating':
                signal_type = SignalType.STRONG_SELL
                signal_confidence = min(0.9, confidence + 0.1)
            else:
                signal_type = SignalType.SELL
                signal_confidence = confidence
        else:
            signal_type = SignalType.HOLD
            signal_confidence = 0.3
        
        # Only generate signal if confidence is sufficient
        if signal_confidence > 0.5:
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=signal_confidence,
                price=0.0,  # Price would come from market data
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={
                    'strategy': 'sentiment_analysis',
                    'sentiment_score': overall_sentiment,
                    'sentiment_label': sentiment_label,
                    'trend': trend,
                    'sources': analysis_results['sources'],
                    'data_points': analysis_results['data_points']
                }
            )
            signals.append(signal)
        
        return signals
    
    def _update_sentiment_trends(self, symbol: str, analysis_results: Dict[str, Any]):
        """Update sentiment trends tracking"""
        self.sentiment_trends[symbol] = {
            'current_sentiment': analysis_results['overall_sentiment'],
            'sentiment_label': analysis_results['sentiment_label'],
            'trend': analysis_results['trend'],
            'confidence': analysis_results['confidence'],
            'last_update': datetime.now(),
            'sources_count': analysis_results['total_sources']
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate sentiment analysis input"""
        return 'market_data' in input_data
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get sentiment analysis summary"""
        return {
            'total_symbols_analyzed': len(self.sentiment_trends),
            'sentiment_trends': self.sentiment_trends,
            'models_available': list(self.sentiment_analyzer.models.keys()),
            'vector_db_available': self.vector_db.index is not None,
            'last_analysis': max(
                [trend['last_update'] for trend in self.sentiment_trends.values()],
                default=datetime.now()
            )
        }

# Example usage and testing
async def main():
    """Example usage of the enhanced sentiment agent"""
    
    # Create sample market data
    sample_data = {
        'AAPL': {
            '1d': {
                'symbol': 'AAPL',
                'timestamp': datetime.now(),
                'open': 150.0,
                'high': 152.5,
                'low': 149.5,
                'close': 151.2,
                'volume': 50000000,
                'timeframe': '1d'
            }
        }
    }
    
    # Initialize sentiment agent
    config = {
        'pinecone_api_key': 'demo-key',
        'newsapi_key': 'demo-key',
        'alpha_vantage_key': 'demo-key'
    }
    
    agent = EnhancedSentimentAgent(config)
    
    # Process sentiment analysis
    input_data = {'market_data': sample_data}
    result = await agent.process(input_data)
    
    print("Sentiment Analysis Results:")
    print(f"Status: {result.status}")
    print(f"Total Signals: {len(result.signals)}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    
    for signal in result.signals:
        print(f"\nSentiment Signal: {signal.symbol}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Sentiment Score: {signal.metadata['sentiment_score']:.2f}")
        print(f"  Sentiment Label: {signal.metadata['sentiment_label']}")
        print(f"  Trend: {signal.metadata['trend']}")
        print(f"  Data Points: {signal.metadata['data_points']}")
    
    # Print sentiment summary
    summary = agent.get_sentiment_summary()
    print(f"\nSentiment Summary:")
    print(f"Symbols Analyzed: {summary['total_symbols_analyzed']}")
    print(f"Models Available: {summary['models_available']}")
    print(f"Vector DB Available: {summary['vector_db_available']}")

if __name__ == "__main__":
    asyncio.run(main())
