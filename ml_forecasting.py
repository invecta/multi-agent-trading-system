"""
Machine Learning Price Forecasting Module
Advanced predictive models for financial price forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MLPriceForecaster:
    """Advanced ML models for price forecasting"""
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0)
        }
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def create_features(self, df, lookback_periods=[5, 10, 20, 50]):
        """Create advanced features for ML models"""
        features_df = df.copy()
        
        # Price-based features
        features_df['returns'] = features_df['Close'].pct_change()
        features_df['log_returns'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
        features_df['price_change'] = features_df['Close'] - features_df['Close'].shift(1)
        
        # Moving averages
        for period in lookback_periods:
            features_df[f'sma_{period}'] = features_df['Close'].rolling(window=period).mean()
            features_df[f'ema_{period}'] = features_df['Close'].ewm(span=period).mean()
            features_df[f'price_to_sma_{period}'] = features_df['Close'] / features_df[f'sma_{period}']
        
        # Volatility features
        features_df['volatility_5'] = features_df['returns'].rolling(window=5).std()
        features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()
        features_df['volatility_ratio'] = features_df['volatility_5'] / features_df['volatility_20']
        
        # Volume features
        features_df['volume_sma_20'] = features_df['Volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_sma_20']
        features_df['price_volume'] = features_df['Close'] * features_df['Volume']
        
        # Technical indicators
        features_df['rsi'] = self.calculate_rsi(features_df['Close'])
        features_df['macd'] = self.calculate_macd(features_df['Close'])
        features_df['bollinger_upper'], features_df['bollinger_lower'] = self.calculate_bollinger_bands(features_df['Close'])
        features_df['bollinger_position'] = (features_df['Close'] - features_df['bollinger_lower']) / (features_df['bollinger_upper'] - features_df['bollinger_lower'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = features_df['Close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['Volume'].shift(lag)
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        
        # Time-based features
        if 'Date' in features_df.columns:
            features_df['day_of_week'] = pd.to_datetime(features_df['Date']).dt.dayofweek
            features_df['month'] = pd.to_datetime(features_df['Date']).dt.month
            features_df['quarter'] = pd.to_datetime(features_df['Date']).dt.quarter
        
        return features_df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def prepare_training_data(self, features_df, target_column='Close', prediction_horizon=1):
        """Prepare data for training"""
        # Select feature columns (exclude target and non-numeric columns)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] 
                          and features_df[col].dtype in ['float64', 'int64']]
        
        # Create target variable (future price)
        features_df['target'] = features_df[target_column].shift(-prediction_horizon)
        
        # Remove rows with NaN values
        clean_df = features_df[feature_columns + ['target']].dropna()
        
        X = clean_df[feature_columns]
        y = clean_df['target']
        
        return X, y, feature_columns
    
    def train_models(self, X, y, feature_columns):
        """Train all ML models"""
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['features'] = scaler
        
        # Train models
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Use scaled data for linear models
            if name in ['LinearRegression', 'Ridge']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.model_performance[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_columns, model.feature_importances_))
        
        return X_test, y_test
    
    def predict_future_prices(self, features_df, feature_columns, horizon_days=5):
        """Predict future prices using the best model"""
        # Find best model based on R² score
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['r2'])
        best_model = self.models[best_model_name]
        
        print(f"Using best model: {best_model_name} (R² = {self.model_performance[best_model_name]['r2']:.4f})")
        
        # Get latest features
        latest_features = features_df[feature_columns].iloc[-1:].copy()
        
        predictions = []
        current_features = latest_features.copy()
        
        for day in range(horizon_days):
            # Scale features if needed
            if best_model_name in ['LinearRegression', 'Ridge']:
                current_features_scaled = self.scalers['features'].transform(current_features)
                pred = best_model.predict(current_features_scaled)[0]
            else:
                pred = best_model.predict(current_features)[0]
            
            predictions.append(pred)
            
            # Update features for next prediction (simplified approach)
            # In practice, you'd need to update all lag features properly
            if day < horizon_days - 1:
                # Shift lag features
                for col in current_features.columns:
                    if 'lag_1' in col:
                        current_features[col] = pred
                    elif 'lag_2' in col:
                        current_features[col] = current_features[col.replace('lag_2', 'lag_1')]
                    elif 'lag_3' in col:
                        current_features[col] = current_features[col.replace('lag_3', 'lag_2')]
        
        return predictions, best_model_name
    
    def generate_forecast_report(self, predictions, model_name, current_price, horizon_days=5):
        """Generate comprehensive forecast report"""
        report = {
            'model_used': model_name,
            'current_price': current_price,
            'forecast_horizon': horizon_days,
            'predictions': predictions,
            'price_changes': [pred - current_price for pred in predictions],
            'percentage_changes': [((pred - current_price) / current_price) * 100 for pred in predictions],
            'model_performance': self.model_performance[model_name],
            'feature_importance': self.feature_importance.get(model_name, {}),
            'forecast_summary': {
                'expected_return': np.mean([((pred - current_price) / current_price) * 100 for pred in predictions]),
                'volatility': np.std([((pred - current_price) / current_price) * 100 for pred in predictions]),
                'max_gain': max([((pred - current_price) / current_price) * 100 for pred in predictions]),
                'max_loss': min([((pred - current_price) / current_price) * 100 for pred in predictions])
            }
        }
        
        return report
    
    def run_complete_forecast(self, df, symbol, horizon_days=5):
        """Run complete forecasting pipeline"""
        print(f"Starting ML forecasting for {symbol}...")
        
        # Create features
        features_df = self.create_features(df)
        
        # Prepare training data
        X, y, feature_columns = self.prepare_training_data(features_df)
        
        if len(X) < 100:  # Need sufficient data
            print("Insufficient data for ML forecasting")
            return None
        
        # Train models
        X_test, y_test = self.train_models(X, y, feature_columns)
        
        # Generate predictions
        current_price = df['Close'].iloc[-1]
        predictions, best_model = self.predict_future_prices(features_df, feature_columns, horizon_days)
        
        # Generate report
        report = self.generate_forecast_report(predictions, best_model, current_price, horizon_days)
        
        print(f"Forecasting completed. Best model: {best_model}")
        print(f"Expected return over {horizon_days} days: {report['forecast_summary']['expected_return']:.2f}%")
        
        return report

def demo_ml_forecasting():
    """Demo function for ML forecasting"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    volumes = np.random.randint(1000000, 5000000, 500)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(500) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(500)) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(500)) * 0.02),
        'Close': prices,
        'Volume': volumes
    })
    
    # Initialize forecaster
    forecaster = MLPriceForecaster()
    
    # Run forecast
    report = forecaster.run_complete_forecast(df, 'DEMO', horizon_days=5)
    
    if report:
        print("\n=== ML FORECASTING REPORT ===")
        print(f"Model: {report['model_used']}")
        print(f"Current Price: ${report['current_price']:.2f}")
        print(f"R² Score: {report['model_performance']['r2']:.4f}")
        print(f"RMSE: {report['model_performance']['rmse']:.2f}")
        print(f"Expected Return: {report['forecast_summary']['expected_return']:.2f}%")
        print(f"Forecast Volatility: {report['forecast_summary']['volatility']:.2f}%")
        
        print("\n5-Day Price Forecast:")
        for i, pred in enumerate(report['predictions']):
            change = report['percentage_changes'][i]
            print(f"Day {i+1}: ${pred:.2f} ({change:+.2f}%)")
        
        print("\nTop 5 Most Important Features:")
        if report['feature_importance']:
            sorted_features = sorted(report['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    demo_ml_forecasting()
