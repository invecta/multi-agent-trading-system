"""
User authentication and portfolio management system
Provides secure user authentication, portfolio management, and user preferences
"""

import hashlib
import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sqlite3
from contextlib import contextmanager
import jwt
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    PREMIUM = "premium"
    STANDARD = "standard"
    BASIC = "basic"

class PortfolioType(Enum):
    """Portfolio type enumeration"""
    GROWTH = "growth"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: datetime
    is_active: bool = True
    preferences: Dict = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

@dataclass
class Portfolio:
    """Portfolio data structure"""
    portfolio_id: str
    user_id: str
    name: str
    portfolio_type: PortfolioType
    created_at: datetime
    last_updated: datetime
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions: Dict = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity

class DatabaseManager:
    """Database management for user and portfolio data"""
    
    def __init__(self, db_path: str = "trading_app.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    preferences TEXT
                )
            """)
            
            # Portfolios table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    portfolio_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    portfolio_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    total_value REAL DEFAULT 0.0,
                    cash_balance REAL DEFAULT 0.0,
                    positions TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            conn.commit()
            
    @contextmanager
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict]:
        """Execute a query and return results"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute an update query and return affected rows"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

class AuthenticationManager:
    """User authentication management"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
        self.session_timeout = timedelta(hours=24)
        
    def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash a password with salt"""
        
        if salt is None:
            salt = secrets.token_hex(16)
            
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return password_hash.hex(), salt
        
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify a password against its hash"""
        
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash
        
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.STANDARD) -> Optional[User]:
        """Create a new user"""
        
        try:
            # Check if user already exists
            existing_user = self.db_manager.execute_query(
                "SELECT user_id FROM users WHERE username = ? OR email = ?",
                (username, email)
            )
            
            if existing_user:
                logger.warning(f"User {username} or email {email} already exists")
                return None
                
            # Create new user
            user_id = secrets.token_hex(16)
            password_hash, salt = self.hash_password(password)
            created_at = datetime.now()
            
            self.db_manager.execute_update(
                """INSERT INTO users (user_id, username, email, password_hash, salt, role, created_at, is_active, preferences)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, username, email, password_hash, salt, role.value, created_at, True, json.dumps({}))
            )
            
            return User(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                created_at=created_at,
                last_login=created_at
            )
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
            
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user"""
        
        try:
            # Get user data
            user_data = self.db_manager.execute_query(
                "SELECT * FROM users WHERE username = ? AND is_active = 1",
                (username,)
            )
            
            if not user_data:
                logger.warning(f"User {username} not found or inactive")
                return None
                
            user_data = user_data[0]
            
            # Verify password
            if not self.verify_password(password, user_data['password_hash'], user_data['salt']):
                logger.warning(f"Invalid password for user {username}")
                return None
                
            # Update last login
            self.db_manager.execute_update(
                "UPDATE users SET last_login = ? WHERE user_id = ?",
                (datetime.now(), user_data['user_id'])
            )
            
            # Create user object
            user = User(
                user_id=user_data['user_id'],
                username=user_data['username'],
                email=user_data['email'],
                role=UserRole(user_data['role']),
                created_at=datetime.fromisoformat(user_data['created_at']),
                last_login=datetime.now(),
                is_active=bool(user_data['is_active']),
                preferences=json.loads(user_data['preferences'] or '{}')
            )
            
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
            
    def create_session(self, user: User) -> str:
        """Create a user session"""
        
        session_id = secrets.token_hex(32)
        created_at = datetime.now()
        expires_at = created_at + self.session_timeout
        
        self.db_manager.execute_update(
            """INSERT INTO user_sessions (session_id, user_id, created_at, expires_at, is_active)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, user.user_id, created_at, expires_at, True)
        )
        
        return session_id
        
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate a user session"""
        
        try:
            # Get session data
            session_data = self.db_manager.execute_query(
                "SELECT * FROM user_sessions WHERE session_id = ? AND is_active = 1",
                (session_id,)
            )
            
            if not session_data:
                return None
                
            session_data = session_data[0]
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                # Deactivate expired session
                self.db_manager.execute_update(
                    "UPDATE user_sessions SET is_active = 0 WHERE session_id = ?",
                    (session_id,)
                )
                return None
                
            # Get user data
            user_data = self.db_manager.execute_query(
                "SELECT * FROM users WHERE user_id = ? AND is_active = 1",
                (session_data['user_id'],)
            )
            
            if not user_data:
                return None
                
            user_data = user_data[0]
            
            # Create user object
            user = User(
                user_id=user_data['user_id'],
                username=user_data['username'],
                email=user_data['email'],
                role=UserRole(user_data['role']),
                created_at=datetime.fromisoformat(user_data['created_at']),
                last_login=datetime.fromisoformat(user_data['last_login'] or user_data['created_at']),
                is_active=bool(user_data['is_active']),
                preferences=json.loads(user_data['preferences'] or '{}')
            )
            
            return user
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
            
    def logout_user(self, session_id: str) -> bool:
        """Logout a user by deactivating their session"""
        
        try:
            affected_rows = self.db_manager.execute_update(
                "UPDATE user_sessions SET is_active = 0 WHERE session_id = ?",
                (session_id,)
            )
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return False

class PortfolioManager:
    """Portfolio management system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def create_portfolio(self, user_id: str, name: str, portfolio_type: PortfolioType, 
                        initial_cash: float = 0.0) -> Optional[Portfolio]:
        """Create a new portfolio"""
        
        try:
            portfolio_id = secrets.token_hex(16)
            created_at = datetime.now()
            
            self.db_manager.execute_update(
                """INSERT INTO portfolios (portfolio_id, user_id, name, portfolio_type, created_at, last_updated, cash_balance)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (portfolio_id, user_id, name, portfolio_type.value, created_at, created_at, initial_cash)
            )
            
            return Portfolio(
                portfolio_id=portfolio_id,
                user_id=user_id,
                name=name,
                portfolio_type=portfolio_type,
                created_at=created_at,
                last_updated=created_at,
                cash_balance=initial_cash
            )
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            return None
            
    def get_user_portfolios(self, user_id: str) -> List[Portfolio]:
        """Get all portfolios for a user"""
        
        try:
            portfolios_data = self.db_manager.execute_query(
                "SELECT * FROM portfolios WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            )
            
            portfolios = []
            for data in portfolios_data:
                portfolio = Portfolio(
                    portfolio_id=data['portfolio_id'],
                    user_id=data['user_id'],
                    name=data['name'],
                    portfolio_type=PortfolioType(data['portfolio_type']),
                    created_at=datetime.fromisoformat(data['created_at']),
                    last_updated=datetime.fromisoformat(data['last_updated']),
                    total_value=data['total_value'],
                    cash_balance=data['cash_balance'],
                    positions=json.loads(data['positions'] or '{}')
                )
                portfolios.append(portfolio)
                
            return portfolios
            
        except Exception as e:
            logger.error(f"Error getting user portfolios: {e}")
            return []
            
    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get a specific portfolio"""
        
        try:
            portfolio_data = self.db_manager.execute_query(
                "SELECT * FROM portfolios WHERE portfolio_id = ?",
                (portfolio_id,)
            )
            
            if not portfolio_data:
                return None
                
            data = portfolio_data[0]
            
            return Portfolio(
                portfolio_id=data['portfolio_id'],
                user_id=data['user_id'],
                name=data['name'],
                portfolio_type=PortfolioType(data['portfolio_type']),
                created_at=datetime.fromisoformat(data['created_at']),
                last_updated=datetime.fromisoformat(data['last_updated']),
                total_value=data['total_value'],
                cash_balance=data['cash_balance'],
                positions=json.loads(data['positions'] or '{}')
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return None
            
    def update_portfolio(self, portfolio: Portfolio) -> bool:
        """Update a portfolio"""
        
        try:
            self.db_manager.execute_update(
                """UPDATE portfolios SET name = ?, portfolio_type = ?, last_updated = ?, 
                   total_value = ?, cash_balance = ?, positions = ? WHERE portfolio_id = ?""",
                (portfolio.name, portfolio.portfolio_type.value, datetime.now(), 
                 portfolio.total_value, portfolio.cash_balance, 
                 json.dumps(portfolio.positions), portfolio.portfolio_id)
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            return False
            
    def add_position(self, portfolio_id: str, symbol: str, quantity: float, 
                    price: float, transaction_type: str = "BUY") -> bool:
        """Add a position to a portfolio"""
        
        try:
            # Get current portfolio
            portfolio = self.get_portfolio(portfolio_id)
            if not portfolio:
                return False
                
            # Create transaction record
            transaction_id = secrets.token_hex(16)
            self.db_manager.execute_update(
                """INSERT INTO transactions (transaction_id, portfolio_id, symbol, transaction_type, quantity, price, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (transaction_id, portfolio_id, symbol, transaction_type, quantity, price, datetime.now())
            )
            
            # Update position
            if symbol in portfolio.positions:
                existing_pos = portfolio.positions[symbol]
                if transaction_type == "BUY":
                    total_quantity = existing_pos['quantity'] + quantity
                    total_cost = (existing_pos['quantity'] * existing_pos['average_price']) + (quantity * price)
                    new_avg_price = total_cost / total_quantity
                    portfolio.positions[symbol] = {
                        'quantity': total_quantity,
                        'average_price': new_avg_price,
                        'current_price': price,
                        'last_updated': datetime.now().isoformat()
                    }
                else:  # SELL
                    portfolio.positions[symbol]['quantity'] -= quantity
                    if portfolio.positions[symbol]['quantity'] <= 0:
                        del portfolio.positions[symbol]
            else:
                if transaction_type == "BUY":
                    portfolio.positions[symbol] = {
                        'quantity': quantity,
                        'average_price': price,
                        'current_price': price,
                        'last_updated': datetime.now().isoformat()
                    }
                    
            # Update cash balance
            if transaction_type == "BUY":
                portfolio.cash_balance -= quantity * price
            else:
                portfolio.cash_balance += quantity * price
                
            # Update portfolio
            return self.update_portfolio(portfolio)
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
            
    def get_portfolio_performance(self, portfolio_id: str) -> Dict:
        """Get portfolio performance metrics"""
        
        try:
            portfolio = self.get_portfolio(portfolio_id)
            if not portfolio:
                return {}
                
            # Calculate performance metrics
            total_market_value = portfolio.cash_balance
            total_cost = 0.0
            
            for symbol, position in portfolio.positions.items():
                market_value = position['quantity'] * position['current_price']
                cost = position['quantity'] * position['average_price']
                total_market_value += market_value
                total_cost += cost
                
            total_return = (total_market_value - total_cost) / total_cost if total_cost > 0 else 0
            
            return {
                'total_value': total_market_value,
                'cash_balance': portfolio.cash_balance,
                'invested_amount': total_cost,
                'total_return': total_return,
                'total_return_percent': total_return * 100,
                'position_count': len(portfolio.positions),
                'last_updated': portfolio.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {}

class UserPreferencesManager:
    """User preferences management"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences"""
        
        try:
            self.db_manager.execute_update(
                "UPDATE users SET preferences = ? WHERE user_id = ?",
                (json.dumps(preferences), user_id)
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
            
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        
        try:
            user_data = self.db_manager.execute_query(
                "SELECT preferences FROM users WHERE user_id = ?",
                (user_id,)
            )
            
            if user_data:
                return json.loads(user_data[0]['preferences'] or '{}')
            return {}
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}

class UserManagementSystem:
    """Complete user management system"""
    
    def __init__(self, db_path: str = "trading_app.db"):
        self.db_manager = DatabaseManager(db_path)
        self.auth_manager = AuthenticationManager(self.db_manager)
        self.portfolio_manager = PortfolioManager(self.db_manager)
        self.preferences_manager = UserPreferencesManager(self.db_manager)
        
    def register_user(self, username: str, email: str, password: str, 
                     role: UserRole = UserRole.STANDARD) -> Optional[User]:
        """Register a new user"""
        
        return self.auth_manager.create_user(username, email, password, role)
        
    def login_user(self, username: str, password: str) -> Tuple[Optional[User], Optional[str]]:
        """Login a user and return user object and session ID"""
        
        user = self.auth_manager.authenticate_user(username, password)
        if user:
            session_id = self.auth_manager.create_session(user)
            return user, session_id
        return None, None
        
    def logout_user(self, session_id: str) -> bool:
        """Logout a user"""
        
        return self.auth_manager.logout_user(session_id)
        
    def get_user_from_session(self, session_id: str) -> Optional[User]:
        """Get user from session ID"""
        
        return self.auth_manager.validate_session(session_id)
        
    def create_user_portfolio(self, user_id: str, name: str, 
                            portfolio_type: PortfolioType, initial_cash: float = 0.0) -> Optional[Portfolio]:
        """Create a portfolio for a user"""
        
        return self.portfolio_manager.create_portfolio(user_id, name, portfolio_type, initial_cash)
        
    def get_user_portfolios(self, user_id: str) -> List[Portfolio]:
        """Get all portfolios for a user"""
        
        return self.portfolio_manager.get_user_portfolios(user_id)
        
    def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user preferences"""
        
        return self.preferences_manager.update_user_preferences(user_id, preferences)
        
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        
        return self.preferences_manager.get_user_preferences(user_id)

# Global instance
user_management_system = UserManagementSystem()
