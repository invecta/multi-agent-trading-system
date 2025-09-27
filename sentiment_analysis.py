"""
Sentiment Analysis Module
News and social media sentiment analysis for trading decisions
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial news and social media"""
    
    def __init__(self):
        self.news_sentiment = {}
        self.social_sentiment = {}
        self.combined_sentiment = {}
        self.sentiment_history = []
        
        # Financial keywords for context
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'strong',
            'positive', 'optimistic', 'outperform', 'upgrade', 'buy', 'outperform',
            'breakthrough', 'milestone', 'record', 'beat', 'exceed', 'strong'
        ]
        
        self.negative_keywords = [
            'bearish', 'decline', 'fall', 'loss', 'weak', 'negative', 'pessimistic',
            'underperform', 'downgrade', 'sell', 'crash', 'plunge', 'miss', 'disappoint',
            'concern', 'risk', 'volatile', 'uncertain', 'challenge', 'struggle'
        ]
        
        self.neutral_keywords = [
            'stable', 'maintain', 'hold', 'neutral', 'unchanged', 'flat', 'steady',
            'consistent', 'moderate', 'balanced', 'mixed', 'uncertain'
        ]
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using multiple methods"""
        if not text or len(text.strip()) == 0:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Keyword-based sentiment
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        neutral_count = sum(1 for word in self.neutral_keywords if word in text)
        
        # Calculate keyword sentiment score
        total_keywords = positive_count + negative_count + neutral_count
        if total_keywords > 0:
            keyword_score = (positive_count - negative_count) / total_keywords
        else:
            keyword_score = 0
        
        # Combine TextBlob and keyword analysis
        combined_score = (polarity * 0.6) + (keyword_score * 0.4)
        
        # Determine sentiment category
        if combined_score > 0.1:
            sentiment = 'positive'
        elif combined_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'keyword_score': keyword_score,
            'combined_score': combined_score,
            'sentiment': sentiment,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'neutral_keywords': neutral_count
        }
    
    def fetch_financial_news(self, symbol, days_back=7):
        """Fetch financial news for a symbol (simulated)"""
        # In a real implementation, you would use APIs like:
        # - Alpha Vantage News API
        # - NewsAPI
        # - Financial Modeling Prep
        # - Yahoo Finance News
        
        # Simulated news data
        news_data = [
            {
                'title': f'{symbol} Reports Strong Q4 Earnings, Beats Expectations',
                'content': f'{symbol} announced better-than-expected quarterly results with revenue growth of 15%. The company showed strong performance across all segments.',
                'source': 'Financial Times',
                'published_at': datetime.now() - timedelta(days=1),
                'url': f'https://example.com/news/{symbol.lower()}-earnings'
            },
            {
                'title': f'Analysts Upgrade {symbol} Price Target Following Positive Outlook',
                'content': f'Multiple analysts have raised their price targets for {symbol} citing improved market conditions and strong fundamentals.',
                'source': 'Reuters',
                'published_at': datetime.now() - timedelta(days=2),
                'url': f'https://example.com/news/{symbol.lower()}-upgrade'
            },
            {
                'title': f'{symbol} Faces Market Volatility Amid Economic Uncertainty',
                'content': f'{symbol} shares experienced increased volatility as investors react to broader market concerns and economic indicators.',
                'source': 'Bloomberg',
                'published_at': datetime.now() - timedelta(days=3),
                'url': f'https://example.com/news/{symbol.lower()}-volatility'
            },
            {
                'title': f'{symbol} Announces New Product Launch, Market Reacts Positively',
                'content': f'{symbol} unveiled its latest product offering, generating positive market response and analyst coverage.',
                'source': 'MarketWatch',
                'published_at': datetime.now() - timedelta(days=4),
                'url': f'https://example.com/news/{symbol.lower()}-product'
            },
            {
                'title': f'Institutional Investors Increase {symbol} Holdings',
                'content': f'Recent filings show that major institutional investors have increased their positions in {symbol}, signaling confidence.',
                'source': 'Seeking Alpha',
                'published_at': datetime.now() - timedelta(days=5),
                'url': f'https://example.com/news/{symbol.lower()}-institutional'
            }
        ]
        
        return news_data
    
    def analyze_news_sentiment(self, symbol, days_back=7):
        """Analyze sentiment of financial news"""
        news_data = self.fetch_financial_news(symbol, days_back)
        
        sentiment_scores = []
        news_analysis = []
        
        for article in news_data:
            # Analyze title and content
            title_sentiment = self.analyze_text_sentiment(article['title'])
            content_sentiment = self.analyze_text_sentiment(article['content'])
            
            # Weighted sentiment (title has more weight)
            combined_sentiment = (title_sentiment['combined_score'] * 0.7 + 
                                content_sentiment['combined_score'] * 0.3)
            
            sentiment_scores.append(combined_sentiment)
            
            news_analysis.append({
                'title': article['title'],
                'source': article['source'],
                'published_at': article['published_at'],
                'title_sentiment': title_sentiment,
                'content_sentiment': content_sentiment,
                'combined_sentiment': combined_sentiment,
                'url': article['url']
            })
        
        # Calculate overall news sentiment
        overall_sentiment = np.mean(sentiment_scores)
        
        self.news_sentiment[symbol] = {
            'overall_sentiment': overall_sentiment,
            'sentiment_category': self._categorize_sentiment(overall_sentiment),
            'article_count': len(news_data),
            'articles': news_analysis,
            'sentiment_distribution': {
                'positive': len([s for s in sentiment_scores if s > 0.1]),
                'negative': len([s for s in sentiment_scores if s < -0.1]),
                'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
            }
        }
        
        return self.news_sentiment[symbol]
    
    def fetch_social_media_sentiment(self, symbol, days_back=7):
        """Fetch social media sentiment (simulated)"""
        # In a real implementation, you would use APIs like:
        # - Twitter API
        # - Reddit API
        # - StockTwits API
        # - Social media sentiment APIs
        
        # Simulated social media data
        social_data = [
            {
                'platform': 'Twitter',
                'content': f'Just bought more {symbol} shares. Strong fundamentals and great growth potential! #stocks #investing',
                'author': 'Trader123',
                'created_at': datetime.now() - timedelta(hours=2),
                'likes': 45,
                'retweets': 12
            },
            {
                'platform': 'Reddit',
                'content': f'{symbol} is looking bearish in the short term. Market conditions are uncertain. Holding for now.',
                'author': 'Investor456',
                'created_at': datetime.now() - timedelta(hours=4),
                'likes': 23,
                'retweets': 3
            },
            {
                'platform': 'StockTwits',
                'content': f'{symbol} breaking resistance levels. Bullish momentum building. Target $200!',
                'author': 'BullTrader',
                'created_at': datetime.now() - timedelta(hours=6),
                'likes': 67,
                'retweets': 18
            },
            {
                'platform': 'Twitter',
                'content': f'Concerned about {symbol} volatility. Market seems overvalued. Taking profits.',
                'author': 'CautiousInvestor',
                'created_at': datetime.now() - timedelta(hours=8),
                'likes': 34,
                'retweets': 7
            },
            {
                'platform': 'Reddit',
                'content': f'{symbol} earnings next week. Expecting positive results based on industry trends.',
                'author': 'EarningsWatcher',
                'created_at': datetime.now() - timedelta(hours=12),
                'likes': 56,
                'retweets': 9
            }
        ]
        
        return social_data
    
    def analyze_social_sentiment(self, symbol, days_back=7):
        """Analyze sentiment of social media posts"""
        social_data = self.fetch_social_media_sentiment(symbol, days_back)
        
        sentiment_scores = []
        social_analysis = []
        
        for post in social_data:
            sentiment = self.analyze_text_sentiment(post['content'])
            
            # Weight by engagement (likes + retweets)
            engagement_weight = min((post['likes'] + post['retweets']) / 100, 1.0)
            weighted_sentiment = sentiment['combined_score'] * (0.5 + engagement_weight * 0.5)
            
            sentiment_scores.append(weighted_sentiment)
            
            social_analysis.append({
                'platform': post['platform'],
                'content': post['content'],
                'author': post['author'],
                'created_at': post['created_at'],
                'engagement': post['likes'] + post['retweets'],
                'sentiment': sentiment,
                'weighted_sentiment': weighted_sentiment
            })
        
        # Calculate overall social sentiment
        overall_sentiment = np.mean(sentiment_scores)
        
        self.social_sentiment[symbol] = {
            'overall_sentiment': overall_sentiment,
            'sentiment_category': self._categorize_sentiment(overall_sentiment),
            'post_count': len(social_data),
            'posts': social_analysis,
            'platform_distribution': {
                'Twitter': len([p for p in social_data if p['platform'] == 'Twitter']),
                'Reddit': len([p for p in social_data if p['platform'] == 'Reddit']),
                'StockTwits': len([p for p in social_data if p['platform'] == 'StockTwits'])
            },
            'sentiment_distribution': {
                'positive': len([s for s in sentiment_scores if s > 0.1]),
                'negative': len([s for s in sentiment_scores if s < -0.1]),
                'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
            }
        }
        
        return self.social_sentiment[symbol]
    
    def _categorize_sentiment(self, score):
        """Categorize sentiment score"""
        if score > 0.2:
            return 'very_positive'
        elif score > 0.1:
            return 'positive'
        elif score > -0.1:
            return 'neutral'
        elif score > -0.2:
            return 'negative'
        else:
            return 'very_negative'
    
    def combine_sentiment_analysis(self, symbol):
        """Combine news and social media sentiment"""
        news_sentiment = self.news_sentiment.get(symbol, {}).get('overall_sentiment', 0)
        social_sentiment = self.social_sentiment.get(symbol, {}).get('overall_sentiment', 0)
        
        # Weighted combination (news 60%, social 40%)
        combined_score = (news_sentiment * 0.6) + (social_sentiment * 0.4)
        
        self.combined_sentiment[symbol] = {
            'combined_score': combined_score,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'sentiment_category': self._categorize_sentiment(combined_score),
            'confidence': self._calculate_confidence(news_sentiment, social_sentiment),
            'trading_signal': self._generate_trading_signal(combined_score)
        }
        
        return self.combined_sentiment[symbol]
    
    def _calculate_confidence(self, news_sentiment, social_sentiment):
        """Calculate confidence in sentiment analysis"""
        # Higher confidence when news and social sentiment align
        alignment = 1 - abs(news_sentiment - social_sentiment)
        
        # Higher confidence with stronger sentiment
        strength = (abs(news_sentiment) + abs(social_sentiment)) / 2
        
        confidence = (alignment * 0.6) + (strength * 0.4)
        return min(confidence, 1.0)
    
    def _generate_trading_signal(self, combined_score):
        """Generate trading signal based on sentiment"""
        if combined_score > 0.3:
            return 'strong_buy'
        elif combined_score > 0.1:
            return 'buy'
        elif combined_score > -0.1:
            return 'hold'
        elif combined_score > -0.3:
            return 'sell'
        else:
            return 'strong_sell'
    
    def run_complete_sentiment_analysis(self, symbol):
        """Run complete sentiment analysis pipeline"""
        print(f"Starting sentiment analysis for {symbol}...")
        
        # Analyze news sentiment
        news_analysis = self.analyze_news_sentiment(symbol)
        
        # Analyze social sentiment
        social_analysis = self.analyze_social_sentiment(symbol)
        
        # Combine sentiment
        combined_analysis = self.combine_sentiment_analysis(symbol)
        
        # Store in history
        self.sentiment_history.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'analysis': combined_analysis
        })
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'news_analysis': news_analysis,
            'social_analysis': social_analysis,
            'combined_analysis': combined_analysis
        }
    
    def get_sentiment_summary(self, symbol):
        """Get sentiment summary for a symbol"""
        if symbol not in self.combined_sentiment:
            return None
        
        analysis = self.combined_sentiment[symbol]
        
        return {
            'symbol': symbol,
            'overall_sentiment': analysis['combined_score'],
            'sentiment_category': analysis['sentiment_category'],
            'trading_signal': analysis['trading_signal'],
            'confidence': analysis['confidence'],
            'news_sentiment': analysis['news_sentiment'],
            'social_sentiment': analysis['social_sentiment']
        }

def demo_sentiment_analysis():
    """Demo function for sentiment analysis"""
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment for a symbol
    result = analyzer.run_complete_sentiment_analysis('AAPL')
    
    print("\n=== SENTIMENT ANALYSIS REPORT ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Timestamp: {result['timestamp']}")
    
    print(f"\nNews Sentiment: {result['news_analysis']['overall_sentiment']:.3f}")
    print(f"Social Sentiment: {result['social_analysis']['overall_sentiment']:.3f}")
    print(f"Combined Sentiment: {result['combined_analysis']['combined_score']:.3f}")
    print(f"Sentiment Category: {result['combined_analysis']['sentiment_category']}")
    print(f"Trading Signal: {result['combined_analysis']['trading_signal']}")
    print(f"Confidence: {result['combined_analysis']['confidence']:.3f}")
    
    print(f"\nNews Articles Analyzed: {result['news_analysis']['article_count']}")
    print(f"Social Posts Analyzed: {result['social_analysis']['post_count']}")
    
    print(f"\nNews Sentiment Distribution:")
    for sentiment, count in result['news_analysis']['sentiment_distribution'].items():
        print(f"  {sentiment}: {count}")
    
    print(f"\nSocial Sentiment Distribution:")
    for sentiment, count in result['social_analysis']['sentiment_distribution'].items():
        print(f"  {sentiment}: {count}")

if __name__ == "__main__":
    demo_sentiment_analysis()
