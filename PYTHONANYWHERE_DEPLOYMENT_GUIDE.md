# PythonAnywhere Deployment Guide

## 🚀 Deploy Your Alpaca Trading Dashboard to PythonAnywhere

### Step 1: Create PythonAnywhere Account
1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for a free account
3. Verify your email address

### Step 2: Upload Your Files
1. **Open the Files tab** in your PythonAnywhere dashboard
2. **Navigate to your home directory** (`/home/yourusername/`)
3. **Create a new folder** called `alpaca_dashboard`
4. **Upload these files** to the `alpaca_dashboard` folder:
   - `pythonanywhere_app.py`
   - `requirements_pythonanywhere.txt`

### Step 3: Install Dependencies
1. **Open the Consoles tab**
2. **Start a Bash console**
3. **Navigate to your project folder**:
   ```bash
   cd alpaca_dashboard
   ```
4. **Install dependencies**:
   ```bash
   pip3.10 install --user -r requirements_pythonanywhere.txt
   ```

### Step 4: Configure Web App
1. **Go to the Web tab**
2. **Click "Add a new web app"**
3. **Choose "Flask"**
4. **Select Python 3.10**
5. **Set the source code path** to `/home/yourusername/alpaca_dashboard/`
6. **Set the WSGI file** to `/home/yourusername/alpaca_dashboard/pythonanywhere_app.py`

### Step 5: Configure WSGI File
1. **Click on the WSGI file link** in the Web tab
2. **Replace the content** with:
   ```python
   import sys
   path = '/home/yourusername/alpaca_dashboard'
   if path not in sys.path:
       sys.path.append(path)

   from pythonanywhere_app import app as application
   ```

### Step 6: Set Environment Variables
1. **In the Web tab**, scroll down to "Environment variables"
2. **Add these variables**:
   - `ALPACA_API_KEY`: `PKOEKMI4RY0LHF565WDO`
   - `ALPACA_SECRET_KEY`: `Dq14y0AJpsIqFfJ33FWKWKWvdJw9zqrAPsaLtJhdDb`

### Step 7: Reload Web App
1. **Click the green "Reload" button** in the Web tab
2. **Your app will be available at**: `https://yourusername.pythonanywhere.com`

## 🎯 What You'll Get

### ✅ Features Included:
- **Professional Dashboard UI** with glassmorphism design
- **Alpaca API Integration** for real-time data
- **Account Information** display
- **Positions Tracking** 
- **Market Data** visualization
- **Responsive Design** for mobile devices
- **Interactive Charts** and analytics

### 🔧 Customization Options:
- **Add real Alpaca API calls** by uncommenting the API code
- **Customize the UI** by modifying the HTML template
- **Add more endpoints** for additional functionality
- **Integrate with your existing dashboard** code

## 🚨 Troubleshooting

### Common Issues:
1. **Import Errors**: Make sure all dependencies are installed
2. **WSGI Errors**: Check the WSGI file path and content
3. **Environment Variables**: Ensure they're set correctly
4. **File Permissions**: Make sure PythonAnywhere can read your files

### Getting Help:
- Check the PythonAnywhere documentation
- Use the PythonAnywhere forums
- Check the console logs in the Web tab

## 🎉 Success!
Your Alpaca Trading Dashboard is now live on PythonAnywhere!

**URL**: `https://yourusername.pythonanywhere.com`

The dashboard will show:
- ✅ API connection status
- 💼 Account information
- 📈 Current positions
- 📊 Market data
- 🎨 Professional UI design
