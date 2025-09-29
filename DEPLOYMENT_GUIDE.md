# 🚀 Live Deployment Guide

## Step 1: Prepare Your Code

1. **Copy your working dashboard to production version:**
   ```bash
   cp WORKING_DASHBOARD_FIXED.py PRODUCTION_DASHBOARD.py
   ```

2. **Update the production file with environment variables:**
   - Replace hardcoded API keys with `os.environ.get()`
   - Set `DEBUG=False` for production
   - Use `HOST='0.0.0.0'` for external access

## Step 2: PythonAnywhere Setup

### 2.1 Upload Files
Upload these files to your PythonAnywhere account:
- `PRODUCTION_DASHBOARD.py`
- `wsgi.py`
- `requirements_production.txt`

### 2.2 Install Dependencies
In the PythonAnywhere console:
```bash
pip3.10 install --user -r requirements_production.txt
```

### 2.3 Configure WSGI
1. Go to Web tab in PythonAnywhere
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select Python 3.10
5. In the WSGI configuration file, replace the content with your `wsgi.py` content
6. Update the path in `wsgi.py` to match your username

### 2.4 Set Environment Variables
In the Web tab, add these environment variables:
```
ALPACA_API_KEY=your-actual-api-key
ALPACA_SECRET_KEY=your-actual-secret-key
SECRET_KEY=your-super-secret-key
DEBUG=False
```

## Step 3: Switch to Live Trading

### 3.1 Update Alpaca Configuration
Change from paper trading to live trading:
```python
ALPACA_BASE_URL = "https://api.alpaca.markets/v2"  # Live trading
# Instead of: "https://paper-api.alpaca.markets/v2"  # Paper trading
```

### 3.2 Update API Keys
Use your live trading API keys from Alpaca dashboard.

## Step 4: Test and Deploy

1. **Test locally first:**
   ```bash
   python PRODUCTION_DASHBOARD.py
   ```

2. **Deploy to PythonAnywhere:**
   - Upload all files
   - Configure WSGI
   - Set environment variables
   - Reload the web app

3. **Verify deployment:**
   - Check your PythonAnywhere URL
   - Test all features
   - Verify live data is working

## Step 5: Security Checklist

- ✅ API keys stored as environment variables
- ✅ DEBUG=False in production
- ✅ Strong SECRET_KEY
- ✅ HTTPS enabled (PythonAnywhere provides this)
- ✅ Input validation on all forms
- ✅ Error handling for API failures

## Step 6: Monitoring

- Monitor your PythonAnywhere logs
- Set up alerts for errors
- Track API usage limits
- Monitor trading performance

## Troubleshooting

### Common Issues:
1. **Import errors:** Check Python path in wsgi.py
2. **API errors:** Verify environment variables
3. **Permission errors:** Check file permissions
4. **Memory issues:** Optimize code for production

### Support:
- PythonAnywhere documentation
- Alpaca API documentation
- Flask deployment guides

## 🎯 Ready for Live Trading!

Once deployed, your dashboard will be live and ready for real trading!
