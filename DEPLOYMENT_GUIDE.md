# ASL Sign Language Recognition - Deployment Guide

## 🚀 Quick Start - Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## ☁️ Heroku Deployment

### Prerequisites
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
- Git installed
- Heroku account created

### Step-by-Step Deployment

#### 1. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit - ASL Flask App"
```

#### 2. Create Heroku App
```bash
heroku create your-asl-app-name
```
Replace `your-asl-app-name` with your desired app name.

#### 3. Configure Buildpacks (Important!)
```bash
heroku buildpacks:add --index 1 heroku/python
heroku buildpacks:add --index 2 https://github.com/heroku/heroku-buildpack-apt
```

#### 4. Create Aptfile for System Dependencies
Create a file named `Aptfile` (no extension) with:
```
libgl1-mesa-glx
libglib2.0-0
```

#### 5. Set Environment Variables
```bash
heroku config:set PYTHONUNBUFFERED=1
heroku config:set WEB_CONCURRENCY=1
```

#### 6. Deploy to Heroku
```bash
git add .
git commit -m "Add deployment configuration"
git push heroku main
```

#### 7. Open Your App
```bash
heroku open
```

### Alternative: Deploy via GitHub

1. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/your-username/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

2. **Connect Heroku to GitHub:**
   - Go to your Heroku Dashboard
   - Select your app → "Deploy" tab
   - Connect to GitHub and select your repository
   - Enable automatic deployments (optional)
   - Click "Deploy Branch"

## 🔧 Configuration Files Explained

### `app.py`
- Main Flask application
- Handles webcam video processing
- Provides REST API endpoints for prediction

### `requirements.txt`
- Python dependencies
- Uses `opencv-python-headless` for server deployment

### `Procfile`
- Tells Heroku how to run your app
- Uses `gunicorn` as WSGI server

### `runtime.txt`
- Specifies Python version for Heroku

### `Aptfile`
- System-level dependencies for OpenCV and MediaPipe

## 🌐 Environment-Specific Notes

### Local Development
- Uses `opencv-python` (with GUI support)
- Debug mode enabled
- Direct camera access

### Production (Heroku)
- Uses `opencv-python-headless` (no GUI)
- Production-ready WSGI server (gunicorn)
- HTTPS required for camera access in browsers

## 📱 Browser Requirements

### Camera Access
- **HTTPS Required:** Modern browsers require HTTPS for camera access
- **Permissions:** Users must grant camera permission
- **Supported Browsers:** Chrome, Firefox, Safari, Edge

### Performance Tips
- Reduce video resolution for better performance
- Adjust processing interval (currently 200ms)
- Consider using WebRTC for real-time streaming

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenCV Import Error
```bash
# Make sure you're using the headless version for deployment
pip install opencv-python-headless
```

#### 2. MediaPipe Issues
```bash
# MediaPipe requires specific system libraries
# Make sure Aptfile includes required dependencies
```

#### 3. Large Slug Size
```bash
# Check slug size
heroku apps:info --app your-app-name

# If too large, consider:
# - Using .slugignore file
# - Removing unnecessary files
```

#### 4. Memory Issues
```bash
# Increase dyno memory
heroku ps:scale web=1:standard-1x
```

#### 5. Camera Not Working
- Ensure HTTPS is enabled
- Check browser permissions
- Verify SSL certificate

### Heroku Logs
```bash
# View real-time logs
heroku logs --tail

# View recent logs
heroku logs --app your-app-name
```

## 🔒 Security Considerations

1. **HTTPS Only:** Camera access requires HTTPS
2. **CORS:** Configure if needed for cross-origin requests
3. **Rate Limiting:** Consider implementing rate limiting
4. **Input Validation:** Validate image data size and format

## 💡 Performance Optimization

### Server-Side
- Use WebSocket for real-time communication
- Implement image compression
- Cache model predictions
- Use Redis for session management

### Client-Side
- Reduce video resolution
- Implement client-side image processing
- Use Web Workers for heavy computations
- Optimize frame rate

## 📊 Monitoring

### Heroku Metrics
```bash
heroku addons:create librato:development
```

### Custom Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🚀 Advanced Deployment Options

### Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

### AWS/GCP Deployment
- Use container services (ECS, Cloud Run)
- Configure load balancers
- Set up auto-scaling
- Use managed databases for session storage

## 📞 Support

If you encounter issues:
1. Check Heroku logs: `heroku logs --tail`
2. Verify all files are committed: `git status`
3. Test locally first: `python app.py`
4. Check browser console for JavaScript errors

---

**Happy Deploying! 🎉**

Your ASL Sign Language Recognition app will help bridge communication gaps and make technology more accessible!
