"""
Social sentiment analysis module with Twitter/Reddit integration
Analyzes social media sentiment for market impact and trading signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import requests
import json
import time
import logging
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaAnalyzer:
    """Social media sentiment analysis engine"""
    
    def __init__(self):
        self.platforms = ['twitter', 'reddit', 'stocktwits', 'news']
        self.sentiment_keywords = {
            'positive': ['bullish', 'buy', 'moon', 'rocket', 'gains', 'profit', 'up', 'rise', 'surge'],
            'negative': ['bearish', 'sell', 'crash', 'drop', 'fall', 'loss', 'down', 'decline', 'plunge'],
            'neutral': ['hold', 'wait', 'stable', 'sideways', 'consolidate']
        }
        
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a text using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'confidence': abs(polarity)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'polarity': 0,
                'subjectivity': 0.5,
                'sentiment': 'neutral',
                'confidence': 0
            }
            
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction (in production, use more sophisticated NLP)
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        
        # Filter for financial/trading keywords
        financial_keywords = [
            'stock', 'price', 'market', 'trading', 'invest', 'portfolio',
            'earnings', 'revenue', 'profit', 'loss', 'bull', 'bear',
            'buy', 'sell', 'hold', 'short', 'long', 'call', 'put'
        ]
        
        return [word for word in words if word in financial_keywords]

class TwitterAnalyzer:
    """Twitter sentiment analysis"""
    
    def __init__(self):
        self.base_url = "https://api.twitter.com/2"
        self.hashtags = ['#stocks', '#trading', '#investing', '#wallstreet']
        
    def get_tweets_about_symbol(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get tweets about a specific symbol (simulated)"""
        # In production, this would use Twitter API
        # For demo purposes, we'll simulate tweet data
        
        tweets = []
        for i in range(count):
            # Simulate tweet content
            tweet_texts = [
                f"{symbol} looking bullish today! Great earnings report",
                f"Thinking about buying more {symbol} shares",
                f"{symbol} price action is concerning, might sell soon",
                f"Long term holder of {symbol}, not worried about short term volatility",
                f"{symbol} breaking resistance levels, momentum building"
            ]
            
            tweet = {
                'id': f"tweet_{i}",
                'text': np.random.choice(tweet_texts),
                'created_at': (datetime.now() - timedelta(hours=np.random.randint(0, 24))).isoformat(),
                'user': f"trader_{i}",
                'retweet_count': np.random.randint(0, 100),
                'like_count': np.random.randint(0, 500),
                'reply_count': np.random.randint(0, 50)
            }
            tweets.append(tweet)
            
        return tweets
        
    def analyze_twitter_sentiment(self, symbol: str) -> Dict:
        """Analyze Twitter sentiment for a symbol"""
        tweets = self.get_tweets_about_symbol(symbol)
        analyzer = SocialMediaAnalyzer()
        
        sentiments = []
        total_engagement = 0
        
        for tweet in tweets:
            sentiment = analyzer.analyze_text_sentiment(tweet['text'])
            sentiments.append(sentiment)
            total_engagement += tweet['retweet_count'] + tweet['like_count'] + tweet['reply_count']
            
        # Calculate weighted sentiment
        if sentiments:
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
            
            # Weight by engagement
            engagement_weights = []
            for tweet in tweets:
                weight = (tweet['retweet_count'] + tweet['like_count'] + tweet['reply_count']) / max(total_engagement, 1)
                engagement_weights.append(weight)
                
            weighted_polarity = np.average([s['polarity'] for s in sentiments], weights=engagement_weights)
            
            return {
                'symbol': symbol,
                'platform': 'twitter',
                'total_tweets': len(tweets),
                'average_polarity': avg_polarity,
                'weighted_polarity': weighted_polarity,
                'average_subjectivity': avg_subjectivity,
                'sentiment': 'positive' if weighted_polarity > 0.1 else 'negative' if weighted_polarity < -0.1 else 'neutral',
                'confidence': abs(weighted_polarity),
                'total_engagement': total_engagement,
                'sample_tweets': tweets[:5]
            }
            
        return {'error': 'No tweets found'}

class RedditAnalyzer:
    """Reddit sentiment analysis"""
    
    def __init__(self):
        self.subreddits = ['wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis']
        
    def get_reddit_posts_about_symbol(self, symbol: str, count: int = 50) -> List[Dict]:
        """Get Reddit posts about a symbol (simulated)"""
        # In production, this would use Reddit API
        
        posts = []
        for i in range(count):
            post_texts = [
                f"DD on {symbol}: Strong fundamentals, undervalued",
                f"{symbol} earnings beat expectations, price target raised",
                f"Concerned about {symbol} management changes",
                f"{symbol} technical analysis shows bullish pattern",
                f"Long term outlook for {symbol} remains positive"
            ]
            
            post = {
                'id': f"post_{i}",
                'title': f"{symbol} Discussion - {np.random.choice(['Analysis', 'News', 'Opinion'])}",
                'text': np.random.choice(post_texts),
                'created_at': (datetime.now() - timedelta(hours=np.random.randint(0, 48))).isoformat(),
                'subreddit': np.random.choice(self.subreddits),
                'score': np.random.randint(0, 1000),
                'num_comments': np.random.randint(0, 200),
                'author': f"redditor_{i}"
            }
            posts.append(post)
            
        return posts
        
    def analyze_reddit_sentiment(self, symbol: str) -> Dict:
        """Analyze Reddit sentiment for a symbol"""
        posts = self.get_reddit_posts_about_symbol(symbol)
        analyzer = SocialMediaAnalyzer()
        
        sentiments = []
        total_score = 0
        
        for post in posts:
            # Analyze both title and text
            title_sentiment = analyzer.analyze_text_sentiment(post['title'])
            text_sentiment = analyzer.analyze_text_sentiment(post['text'])
            
            # Combine sentiments
            combined_polarity = (title_sentiment['polarity'] + text_sentiment['polarity']) / 2
            combined_subjectivity = (title_sentiment['subjectivity'] + text_sentiment['subjectivity']) / 2
            
            sentiments.append({
                'polarity': combined_polarity,
                'subjectivity': combined_subjectivity,
                'sentiment': 'positive' if combined_polarity > 0.1 else 'negative' if combined_polarity < -0.1 else 'neutral'
            })
            
            total_score += post['score']
            
        if sentiments:
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
            
            # Weight by Reddit score
            score_weights = [post['score'] / max(total_score, 1) for post in posts]
            weighted_polarity = np.average([s['polarity'] for s in sentiments], weights=score_weights)
            
            return {
                'symbol': symbol,
                'platform': 'reddit',
                'total_posts': len(posts),
                'average_polarity': avg_polarity,
                'weighted_polarity': weighted_polarity,
                'average_subjectivity': avg_subjectivity,
                'sentiment': 'positive' if weighted_polarity > 0.1 else 'negative' if weighted_polarity < -0.1 else 'neutral',
                'confidence': abs(weighted_polarity),
                'total_score': total_score,
                'sample_posts': posts[:5]
            }
            
        return {'error': 'No posts found'}

class StockTwitsAnalyzer:
    """StockTwits sentiment analysis"""
    
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"
        
    def get_stocktwits_messages(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get StockTwits messages about a symbol (simulated)"""
        # In production, this would use StockTwits API
        
        messages = []
        for i in range(count):
            message_texts = [
                f"${symbol} breaking out! Target $200",
                f"${symbol} support holding strong, bullish",
                f"${symbol} earnings disappointment, selling",
                f"${symbol} long term play, accumulating on dips",
                f"${symbol} technical setup looks good"
            ]
            
            message = {
                'id': f"message_{i}",
                'body': np.random.choice(message_texts),
                'created_at': (datetime.now() - timedelta(minutes=np.random.randint(0, 1440))).isoformat(),
                'user': f"trader_{i}",
                'likes': np.random.randint(0, 50),
                'reshares': np.random.randint(0, 20)
            }
            messages.append(message)
            
        return messages
        
    def analyze_stocktwits_sentiment(self, symbol: str) -> Dict:
        """Analyze StockTwits sentiment for a symbol"""
        messages = self.get_stocktwits_messages(symbol)
        analyzer = SocialMediaAnalyzer()
        
        sentiments = []
        total_engagement = 0
        
        for message in messages:
            sentiment = analyzer.analyze_text_sentiment(message['body'])
            sentiments.append(sentiment)
            total_engagement += message['likes'] + message['reshares']
            
        if sentiments:
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
            
            # Weight by engagement
            engagement_weights = []
            for message in messages:
                weight = (message['likes'] + message['reshares']) / max(total_engagement, 1)
                engagement_weights.append(weight)
                
            weighted_polarity = np.average([s['polarity'] for s in sentiments], weights=engagement_weights)
            
            return {
                'symbol': symbol,
                'platform': 'stocktwits',
                'total_messages': len(messages),
                'average_polarity': avg_polarity,
                'weighted_polarity': weighted_polarity,
                'average_subjectivity': avg_subjectivity,
                'sentiment': 'positive' if weighted_polarity > 0.1 else 'negative' if weighted_polarity < -0.1 else 'neutral',
                'confidence': abs(weighted_polarity),
                'total_engagement': total_engagement,
                'sample_messages': messages[:5]
            }
            
        return {'error': 'No messages found'}

class NewsSentimentAnalyzer:
    """News sentiment analysis"""
    
    def __init__(self):
        self.news_sources = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance']
        
    def get_news_about_symbol(self, symbol: str, count: int = 20) -> List[Dict]:
        """Get news articles about a symbol (simulated)"""
        # In production, this would use news APIs
        
        articles = []
        for i in range(count):
            headlines = [
                f"{symbol} Reports Strong Q4 Earnings, Beats Expectations",
                f"{symbol} Stock Rises on Positive Analyst Upgrade",
                f"{symbol} Faces Regulatory Challenges, Stock Declines",
                f"{symbol} Announces New Product Launch, Market Responds Positively",
                f"{symbol} CEO Resigns, Uncertainty Weighs on Stock"
            ]
            
            article = {
                'id': f"article_{i}",
                'headline': np.random.choice(headlines),
                'summary': f"Detailed analysis of {symbol} recent performance and outlook",
                'published_at': (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat(),
                'source': np.random.choice(self.news_sources),
                'url': f"https://example.com/news/{symbol}_{i}",
                'sentiment_score': np.random.uniform(-1, 1)
            }
            articles.append(article)
            
        return articles
        
    def analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment for a symbol"""
        articles = self.get_news_about_symbol(symbol)
        analyzer = SocialMediaAnalyzer()
        
        sentiments = []
        
        for article in articles:
            # Analyze headline and summary
            headline_sentiment = analyzer.analyze_text_sentiment(article['headline'])
            summary_sentiment = analyzer.analyze_text_sentiment(article['summary'])
            
            # Combine sentiments
            combined_polarity = (headline_sentiment['polarity'] + summary_sentiment['polarity']) / 2
            combined_subjectivity = (headline_sentiment['subjectivity'] + summary_sentiment['subjectivity']) / 2
            
            sentiments.append({
                'polarity': combined_polarity,
                'subjectivity': combined_subjectivity,
                'sentiment': 'positive' if combined_polarity > 0.1 else 'negative' if combined_polarity < -0.1 else 'neutral'
            })
            
        if sentiments:
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
            
            return {
                'symbol': symbol,
                'platform': 'news',
                'total_articles': len(articles),
                'average_polarity': avg_polarity,
                'average_subjectivity': avg_subjectivity,
                'sentiment': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral',
                'confidence': abs(avg_polarity),
                'sample_articles': articles[:5]
            }
            
        return {'error': 'No articles found'}

class SocialSentimentAggregator:
    """Aggregates sentiment from multiple social media platforms"""
    
    def __init__(self):
        self.twitter_analyzer = TwitterAnalyzer()
        self.reddit_analyzer = RedditAnalyzer()
        self.stocktwits_analyzer = StockTwitsAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        
    def analyze_symbol_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment across all platforms for a symbol"""
        
        results = {}
        
        # Twitter analysis
        try:
            results['twitter'] = self.twitter_analyzer.analyze_twitter_sentiment(symbol)
        except Exception as e:
            logger.error(f"Twitter analysis error: {e}")
            results['twitter'] = {'error': str(e)}
            
        # Reddit analysis
        try:
            results['reddit'] = self.reddit_analyzer.analyze_reddit_sentiment(symbol)
        except Exception as e:
            logger.error(f"Reddit analysis error: {e}")
            results['reddit'] = {'error': str(e)}
            
        # StockTwits analysis
        try:
            results['stocktwits'] = self.stocktwits_analyzer.analyze_stocktwits_sentiment(symbol)
        except Exception as e:
            logger.error(f"StockTwits analysis error: {e}")
            results['stocktwits'] = {'error': str(e)}
            
        # News analysis
        try:
            results['news'] = self.news_analyzer.analyze_news_sentiment(symbol)
        except Exception as e:
            logger.error(f"News analysis error: {e}")
            results['news'] = {'error': str(e)}
            
        # Aggregate results
        aggregated = self._aggregate_sentiment_results(results)
        aggregated['platform_results'] = results
        
        return aggregated
        
    def _aggregate_sentiment_results(self, results: Dict) -> Dict:
        """Aggregate sentiment results from all platforms"""
        
        valid_results = []
        platform_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'stocktwits': 0.25,
            'news': 0.2
        }
        
        for platform, result in results.items():
            if 'error' not in result and 'weighted_polarity' in result:
                valid_results.append({
                    'platform': platform,
                    'polarity': result['weighted_polarity'],
                    'confidence': result['confidence'],
                    'weight': platform_weights.get(platform, 0.25)
                })
            elif 'error' not in result and 'average_polarity' in result:
                valid_results.append({
                    'platform': platform,
                    'polarity': result['average_polarity'],
                    'confidence': result['confidence'],
                    'weight': platform_weights.get(platform, 0.25)
                })
                
        if not valid_results:
            return {
                'symbol': 'unknown',
                'overall_sentiment': 'neutral',
                'overall_polarity': 0,
                'overall_confidence': 0,
                'platform_count': 0
            }
            
        # Calculate weighted average
        total_weight = sum(r['weight'] for r in valid_results)
        weighted_polarity = sum(r['polarity'] * r['weight'] for r in valid_results) / total_weight
        weighted_confidence = sum(r['confidence'] * r['weight'] for r in valid_results) / total_weight
        
        # Determine overall sentiment
        if weighted_polarity > 0.1:
            overall_sentiment = 'positive'
        elif weighted_polarity < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
            
        return {
            'symbol': valid_results[0]['platform'] if valid_results else 'unknown',
            'overall_sentiment': overall_sentiment,
            'overall_polarity': weighted_polarity,
            'overall_confidence': weighted_confidence,
            'platform_count': len(valid_results),
            'platform_breakdown': valid_results
        }
        
    def create_sentiment_chart(self, symbol: str) -> go.Figure:
        """Create sentiment visualization chart"""
        
        sentiment_data = self.analyze_symbol_sentiment(symbol)
        
        if 'platform_results' not in sentiment_data:
            return go.Figure()
            
        platforms = []
        polarities = []
        confidences = []
        
        for platform, result in sentiment_data['platform_results'].items():
            if 'error' not in result:
                platforms.append(platform.title())
                if 'weighted_polarity' in result:
                    polarities.append(result['weighted_polarity'])
                else:
                    polarities.append(result['average_polarity'])
                confidences.append(result['confidence'])
                
        if not platforms:
            return go.Figure()
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} Sentiment by Platform', 'Sentiment Confidence'),
            vertical_spacing=0.1
        )
        
        # Sentiment bar chart
        colors = ['green' if p > 0.1 else 'red' if p < -0.1 else 'gray' for p in polarities]
        
        fig.add_trace(
            go.Bar(
                x=platforms,
                y=polarities,
                name='Sentiment',
                marker_color=colors,
                text=[f'{p:.2f}' for p in polarities],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Confidence bar chart
        fig.add_trace(
            go.Bar(
                x=platforms,
                y=confidences,
                name='Confidence',
                marker_color='lightblue',
                text=[f'{c:.2f}' for c in confidences],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} Social Media Sentiment Analysis',
            height=600,
            template='plotly_dark'
        )
        
        return fig

# Global instances
social_analyzer = SocialMediaAnalyzer()
sentiment_aggregator = SocialSentimentAggregator()
