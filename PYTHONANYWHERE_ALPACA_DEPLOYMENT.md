# Alpaca Enhanced Dashboard - PythonAnywhere Deployment Guide

## 🚀 Deploy Your Alpaca Trading Dashboard Online

### Step 1: Upload Files to PythonAnywhere

Upload these files to your PythonAnywhere home directory (`/home/hindaouihani/`):

1. **`pythonanywhere_alpaca_dashboard.py`** - Main dashboard application
2. **`requirements_alpaca.txt`** - Dependencies 
3. **`pythonanywhere_alpaca_wsgi.py`** - WSGI configuration

### Step 2: Install Dependencies

In PythonAnywhere **Console**:

```bash
pip3.10 install --user -r requirements_alpaca.txt
```

### Step 3: Configure Web App

1. Go to **Web** tab in PythonAnywhere dashboard
2. Click **"Add a new web app"** 
3. Choose **"Manual configuration"**
4. Select **Python 3.10**
5. Click **"Next"**

### Step 4: Configure WSGI File

1. In the **WSGI configuration file** section, click **"Create"**
2. Replace all content with:

```python
import sys
path = '/home/hindaouihani'
if path not in sys.path:
    sys.path.append(path)

from pythonanywhere_alpaca_dashboard import application
```

3. Click **"Save"**

### Step 5: Restart Web App

1. In the **Web** tab, click **"Reload"**
2. Your dashboard will be available at: **https://hindaouihani.pythonanywhere.com/**

## ✨ Your Dashboard Features

### 📊 **Alpaca Integration**
- **Live Connection Status** - Shows if connected to your Alpaca paper account
- **Real Account Data** - Your actual buying power, portfolio value, cash balance
- **Live Positions** - Current holdings with profit/loss

### 🔬 **Strategy Backtest** 
- **Moving Average Crossover** (20/50 SMA)
- **Interactive Charts** - Price with trade markers
- **Performance Metrics** - Total return, trades, final value
- **Multiple Symbols** - AAPL, GOOGL, TSLA, MSFT, AMZN, NVDA

### 🎨 **Professional Design**
- **Beautiful Gradient Background** - Like your original HTML version
- **Responsive Cards** - Clean, modern interface
- **Color-Coded Positions** - Green for profit, red for loss

## 🔧 Troubleshooting

### If Dashboard Won't Load:
1. Check **Logs** in Web tab for errors
2. Ensure WSGI file points to correct application name
3. Verify dependencies are installed

### If Alpaca Connection Fails:
- Dashboard still works with simulated data
- Check API keys in your environment variables
- Connection status shows at top of dashboard

### If Charts Don't Load:
- Ensure plotly is properly installed
- Check browser console for JavaScript errors

## 🌐 Your Live Dashboard

**Access:** https://hindaouihani.pythonanywhere.com/

This dashboard combines:
- ✅ **Your HTML design** aesthetics
- ✅ **Real Alpaca data** from your paper account  
- ✅ **Working strategy** that generates actual trades
- ✅ **Online accessibility** for sharing and demo

## 💡 Next Steps

Once deployed, you can:
1. **Share the URL** with others
2. **Add to your portfolio** as a live demo
3. **Extend features** with additional trading strategies
4. **Integrate real-time data** feeds

Your professional trading dashboard is now live and accessible worldwide! 🎉
