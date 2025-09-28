# 🚀 Alpaca Trading Dashboard - Cloud Deployment Guide

## Option 1: Render.com (Recommended - Free Tier Available)

### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Connect your GitHub repository

### Step 2: Deploy to Render
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Use these settings:
   - **Name**: `alpaca-trading-dashboard`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements_render.txt`
   - **Start Command**: `gunicorn alpaca_trading_app:app`
   - **Plan**: Free

### Step 3: Set Environment Variables
In Render dashboard, go to Environment tab and add:
- `ALPACA_API_KEY`: `PKOEKMI4RY0LHF565WDO`
- `ALPACA_SECRET_KEY`: `Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb`

### Step 4: Deploy
Click "Create Web Service" and wait for deployment.

---

## Option 2: Heroku (Alternative)

### Step 1: Install Heroku CLI
```bash
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Login and Create App
```bash
heroku login
heroku create your-alpaca-dashboard
```

### Step 3: Set Environment Variables
```bash
heroku config:set ALPACA_API_KEY=PKOEKMI4RY0LHF565WDO
heroku config:set ALPACA_SECRET_KEY=Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb
```

### Step 4: Deploy
```bash
git add .
git commit -m "Deploy Alpaca trading dashboard"
git push heroku main
```

---

## Option 3: PythonAnywhere (Simple)

### Step 1: Create Account
1. Go to [pythonanywhere.com](https://pythonanywhere.com)
2. Sign up for free account

### Step 2: Upload Files
1. Upload `alpaca_trading_app.py` and `requirements_render.txt`
2. Install dependencies in Bash console:
```bash
pip3.10 install --user -r requirements_render.txt
```

### Step 3: Create Web App
1. Go to Web tab
2. Create new web app
3. Choose Flask
4. Set path to your `alpaca_trading_app.py`

---

## Option 4: Vercel (For Static/API)

### Step 1: Install Vercel CLI
```bash
npm i -g vercel
```

### Step 2: Deploy
```bash
vercel --prod
```

---

## Features Included

✅ **Real-time Alpaca API Integration**
- Account information
- Market data fetching
- Position tracking

✅ **Interactive Charts**
- Candlestick price charts
- Plotly.js integration

✅ **Responsive Design**
- Mobile-friendly interface
- Professional styling

✅ **API Endpoints**
- `/api/account` - Account details
- `/api/positions` - Current positions

---

## Quick Start Commands

```bash
# Test locally
python alpaca_trading_app.py

# Deploy to Render (after connecting GitHub)
# Just push to GitHub - Render auto-deploys

# Deploy to Heroku
git push heroku main

# Deploy to PythonAnywhere
# Upload files via web interface
```

---

## Troubleshooting

### Common Issues:
1. **Alpaca Connection Failed**: Check API keys
2. **Import Errors**: Ensure all dependencies installed
3. **Port Issues**: Use `PORT` environment variable

### Debug Commands:
```bash
# Check Alpaca connection
python -c "from alpaca.trading.client import TradingClient; print('Alpaca OK')"

# Test Flask app
curl http://localhost:5000

# Check environment variables
echo $ALPACA_API_KEY
```

---

## Your Alpaca Demo Account Details
- **API Key**: PKOEKMI4RY0LHF565WDO
- **Secret**: Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb
- **Base URL**: https://paper-api.alpaca.markets
- **Account**: PA3TE0S55RX1 (Paper Trading)
