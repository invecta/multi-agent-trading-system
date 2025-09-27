"""
Real-time Alerts Module
Notification system for trading signals and market events
"""

import smtplib
import json
import time
import threading
from datetime import datetime, timedelta
try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError:
    # Fallback for Python 3.13
    from email.mime.text import MimeText as MIMEText
    from email.mime.multipart import MimeMultipart as MIMEMultipart
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

class AlertManager:
    """Real-time alert system for trading signals"""
    
    def __init__(self, config_file=None):
        self.alerts = []
        self.alert_history = []
        self.running = False
        self.alert_thread = None
        self.config = self.load_config(config_file)
        self.alert_callbacks = {}
        
        # Alert types
        self.alert_types = {
            'price_breakout': 'Price Breakout',
            'volume_spike': 'Volume Spike',
            'technical_signal': 'Technical Signal',
            'sentiment_change': 'Sentiment Change',
            'portfolio_alert': 'Portfolio Alert',
            'risk_alert': 'Risk Alert',
            'ml_prediction': 'ML Prediction',
            'news_alert': 'News Alert'
        }
        
        # Alert channels
        self.channels = {
            'email': self.send_email_alert,
            'console': self.send_console_alert,
            'webhook': self.send_webhook_alert,
            'file': self.send_file_alert
        }
    
    def load_config(self, config_file):
        """Load alert configuration"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from_email': '',
                'to_emails': []
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'headers': {'Content-Type': 'application/json'}
            },
            'file': {
                'enabled': True,
                'path': 'alerts.log'
            },
            'console': {
                'enabled': True
            },
            'alert_settings': {
                'check_interval': 60,  # seconds
                'max_alerts_per_hour': 10,
                'alert_cooldown': 300  # seconds
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except FileNotFoundError:
                print(f"Config file {config_file} not found, using defaults")
        
        return default_config
    
    def add_alert(self, alert_type: str, symbol: str, message: str, 
                  priority: str = 'medium', channels: List[str] = None, 
                  conditions: Dict = None, callback: Callable = None):
        """Add a new alert"""
        if channels is None:
            channels = ['console', 'file']
        
        alert = {
            'id': len(self.alerts) + 1,
            'type': alert_type,
            'symbol': symbol,
            'message': message,
            'priority': priority,
            'channels': channels,
            'conditions': conditions or {},
            'callback': callback,
            'created_at': datetime.now(),
            'triggered': False,
            'triggered_at': None,
            'cooldown_until': None
        }
        
        self.alerts.append(alert)
        return alert['id']
    
    def remove_alert(self, alert_id: int):
        """Remove an alert"""
        self.alerts = [alert for alert in self.alerts if alert['id'] != alert_id]
    
    def check_alert_conditions(self, alert: Dict, current_data: Dict) -> bool:
        """Check if alert conditions are met"""
        conditions = alert['conditions']
        
        if not conditions:
            return True
        
        symbol = alert['symbol']
        
        # Price conditions
        if 'price_above' in conditions:
            if current_data.get('price', 0) <= conditions['price_above']:
                return False
        
        if 'price_below' in conditions:
            if current_data.get('price', 0) >= conditions['price_below']:
                return False
        
        # Volume conditions
        if 'volume_above' in conditions:
            if current_data.get('volume', 0) <= conditions['volume_above']:
                return False
        
        # Technical indicator conditions
        if 'rsi_above' in conditions:
            if current_data.get('rsi', 50) <= conditions['rsi_above']:
                return False
        
        if 'rsi_below' in conditions:
            if current_data.get('rsi', 50) >= conditions['rsi_below']:
                return False
        
        # Sentiment conditions
        if 'sentiment_above' in conditions:
            if current_data.get('sentiment', 0) <= conditions['sentiment_above']:
                return False
        
        if 'sentiment_below' in conditions:
            if current_data.get('sentiment', 0) >= conditions['sentiment_below']:
                return False
        
        # Portfolio conditions
        if 'portfolio_return_above' in conditions:
            if current_data.get('portfolio_return', 0) <= conditions['portfolio_return_above']:
                return False
        
        if 'portfolio_return_below' in conditions:
            if current_data.get('portfolio_return', 0) >= conditions['portfolio_return_below']:
                return False
        
        return True
    
    def trigger_alert(self, alert: Dict, current_data: Dict):
        """Trigger an alert"""
        if alert['triggered'] and alert['cooldown_until']:
            if datetime.now() < alert['cooldown_until']:
                return  # Still in cooldown period
        
        # Check rate limiting
        if self.is_rate_limited():
            return
        
        # Update alert status
        alert['triggered'] = True
        alert['triggered_at'] = datetime.now()
        alert['cooldown_until'] = datetime.now() + timedelta(
            seconds=self.config['alert_settings']['alert_cooldown']
        )
        
        # Create alert message
        alert_message = self.create_alert_message(alert, current_data)
        
        # Send to configured channels
        for channel in alert['channels']:
            if channel in self.channels:
                try:
                    self.channels[channel](alert, alert_message, current_data)
                except Exception as e:
                    print(f"Error sending alert via {channel}: {e}")
        
        # Execute callback if provided
        if alert['callback']:
            try:
                alert['callback'](alert, current_data)
            except Exception as e:
                print(f"Error executing alert callback: {e}")
        
        # Add to history
        self.alert_history.append({
            'alert_id': alert['id'],
            'type': alert['type'],
            'symbol': alert['symbol'],
            'message': alert_message,
            'priority': alert['priority'],
            'triggered_at': datetime.now(),
            'data': current_data
        })
        
        print(f"ALERT TRIGGERED: {alert['type']} for {alert['symbol']} - {alert['message']}")
    
    def create_alert_message(self, alert: Dict, current_data: Dict) -> str:
        """Create formatted alert message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        symbol = alert['symbol']
        message = alert['message']
        priority = alert['priority'].upper()
        
        # Add current data to message
        data_info = []
        if 'price' in current_data:
            data_info.append(f"Price: ${current_data['price']:.2f}")
        if 'volume' in current_data:
            data_info.append(f"Volume: {current_data['volume']:,}")
        if 'rsi' in current_data:
            data_info.append(f"RSI: {current_data['rsi']:.1f}")
        if 'sentiment' in current_data:
            data_info.append(f"Sentiment: {current_data['sentiment']:.3f}")
        
        data_str = " | ".join(data_info) if data_info else ""
        
        alert_message = f"""
🚨 TRADING ALERT - {priority} PRIORITY 🚨

Timestamp: {timestamp}
Symbol: {symbol}
Type: {self.alert_types.get(alert['type'], alert['type'])}
Message: {message}

Current Data: {data_str}

---
Multi-Agent Trading System
        """.strip()
        
        return alert_message
    
    def send_email_alert(self, alert: Dict, message: str, current_data: Dict):
        """Send email alert"""
        if not self.config['email']['enabled']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_email']
            msg['To'] = ', '.join(self.config['email']['to_emails'])
            msg['Subject'] = f"Trading Alert: {alert['symbol']} - {alert['type']}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['username'], self.config['email']['password'])
            
            text = msg.as_string()
            server.sendmail(self.config['email']['from_email'], self.config['email']['to_emails'], text)
            server.quit()
            
            print(f"Email alert sent for {alert['symbol']}")
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def send_console_alert(self, alert: Dict, message: str, current_data: Dict):
        """Send console alert"""
        if not self.config['console']['enabled']:
            return
        
        print("\n" + "="*60)
        print(message)
        print("="*60 + "\n")
    
    def send_webhook_alert(self, alert: Dict, message: str, current_data: Dict):
        """Send webhook alert"""
        if not self.config['webhook']['enabled']:
            return
        
        try:
            payload = {
                'alert': {
                    'id': alert['id'],
                    'type': alert['type'],
                    'symbol': alert['symbol'],
                    'message': message,
                    'priority': alert['priority'],
                    'timestamp': datetime.now().isoformat()
                },
                'data': current_data
            }
            
            response = requests.post(
                self.config['webhook']['url'],
                json=payload,
                headers=self.config['webhook']['headers'],
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Webhook alert sent for {alert['symbol']}")
            else:
                print(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
    
    def send_file_alert(self, alert: Dict, message: str, current_data: Dict):
        """Send file alert"""
        if not self.config['file']['enabled']:
            return
        
        try:
            with open(self.config['file']['path'], 'a') as f:
                f.write(f"\n{datetime.now().isoformat()} - {message}\n")
                f.write(f"Data: {json.dumps(current_data, indent=2)}\n")
                f.write("-" * 80 + "\n")
            
            print(f"File alert logged for {alert['symbol']}")
            
        except Exception as e:
            print(f"Failed to log file alert: {e}")
    
    def is_rate_limited(self) -> bool:
        """Check if we're rate limited"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        recent_alerts = [alert for alert in self.alert_history 
                        if alert['triggered_at'] > hour_ago]
        
        return len(recent_alerts) >= self.config['alert_settings']['max_alerts_per_hour']
    
    def start_monitoring(self):
        """Start the alert monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.alert_thread = threading.Thread(target=self._monitor_alerts, daemon=True)
        self.alert_thread.start()
        print("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop the alert monitoring thread"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join()
        print("Alert monitoring stopped")
    
    def _monitor_alerts(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check all active alerts
                for alert in self.alerts:
                    if not alert['triggered'] or alert['cooldown_until'] is None:
                        # Get current data for the symbol
                        current_data = self.get_current_data(alert['symbol'])
                        
                        if current_data and self.check_alert_conditions(alert, current_data):
                            self.trigger_alert(alert, current_data)
                
                # Sleep for the check interval
                time.sleep(self.config['alert_settings']['check_interval'])
                
            except Exception as e:
                print(f"Error in alert monitoring: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def get_current_data(self, symbol: str) -> Dict:
        """Get current market data for a symbol (simulated)"""
        # In a real implementation, this would fetch live data
        # For demo purposes, we'll simulate some data
        
        np.random.seed(hash(symbol) % 2**32)
        
        # Simulate current market data
        base_price = 150 + np.random.randn() * 10
        current_volume = 1000000 + np.random.randint(0, 5000000)
        rsi = 30 + np.random.rand() * 40  # RSI between 30-70
        sentiment = -0.5 + np.random.rand() * 1.0  # Sentiment between -0.5 and 0.5
        
        return {
            'symbol': symbol,
            'price': base_price,
            'volume': current_volume,
            'rsi': rsi,
            'sentiment': sentiment,
            'timestamp': datetime.now()
        }
    
    def get_alert_history(self, symbol: str = None, alert_type: str = None, 
                         hours_back: int = 24) -> List[Dict]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_history = [
            alert for alert in self.alert_history
            if alert['triggered_at'] > cutoff_time
        ]
        
        if symbol:
            filtered_history = [alert for alert in filtered_history if alert['symbol'] == symbol]
        
        if alert_type:
            filtered_history = [alert for alert in filtered_history if alert['type'] == alert_type]
        
        return sorted(filtered_history, key=lambda x: x['triggered_at'], reverse=True)
    
    def get_alert_summary(self) -> Dict:
        """Get alert system summary"""
        total_alerts = len(self.alerts)
        active_alerts = len([alert for alert in self.alerts if not alert['triggered']])
        triggered_alerts = len([alert for alert in self.alerts if alert['triggered']])
        
        # Recent activity
        recent_alerts = self.get_alert_history(hours_back=24)
        
        # Alert type distribution
        type_distribution = {}
        for alert in self.alerts:
            alert_type = alert['type']
            type_distribution[alert_type] = type_distribution.get(alert_type, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_alerts': triggered_alerts,
            'recent_alerts_24h': len(recent_alerts),
            'type_distribution': type_distribution,
            'monitoring_active': self.running,
            'rate_limited': self.is_rate_limited()
        }

def demo_real_time_alerts():
    """Demo function for real-time alerts"""
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Add some sample alerts
    alert_manager.add_alert(
        alert_type='price_breakout',
        symbol='AAPL',
        message='AAPL price broke above resistance level',
        priority='high',
        channels=['console', 'file'],
        conditions={'price_above': 160}
    )
    
    alert_manager.add_alert(
        alert_type='volume_spike',
        symbol='MSFT',
        message='MSFT volume spike detected',
        priority='medium',
        channels=['console'],
        conditions={'volume_above': 2000000}
    )
    
    alert_manager.add_alert(
        alert_type='technical_signal',
        symbol='GOOGL',
        message='GOOGL RSI oversold condition',
        priority='low',
        channels=['console'],
        conditions={'rsi_below': 30}
    )
    
    alert_manager.add_alert(
        alert_type='sentiment_change',
        symbol='TSLA',
        message='TSLA sentiment turned positive',
        priority='medium',
        channels=['console'],
        conditions={'sentiment_above': 0.3}
    )
    
    # Start monitoring
    alert_manager.start_monitoring()
    
    print("\n=== REAL-TIME ALERTS DEMO ===")
    print("Alert monitoring started. Alerts will trigger based on simulated market conditions.")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        # Monitor for a few minutes
        time.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        pass
    finally:
        alert_manager.stop_monitoring()
        
        # Show summary
        summary = alert_manager.get_alert_summary()
        print("\n=== ALERT SUMMARY ===")
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Active Alerts: {summary['active_alerts']}")
        print(f"Triggered Alerts: {summary['triggered_alerts']}")
        print(f"Recent Alerts (24h): {summary['recent_alerts_24h']}")
        print(f"Monitoring Active: {summary['monitoring_active']}")
        
        # Show recent alerts
        recent_alerts = alert_manager.get_alert_history(hours_back=1)
        if recent_alerts:
            print("\n=== RECENT ALERTS ===")
            for alert in recent_alerts[:5]:  # Show last 5
                print(f"{alert['triggered_at']}: {alert['type']} - {alert['symbol']} - {alert['message']}")

if __name__ == "__main__":
    demo_real_time_alerts()
