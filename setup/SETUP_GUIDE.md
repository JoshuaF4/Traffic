# TrafficAI Pro - Complete Setup Guide

## 🎯 Overview

TrafficAI Pro is a Flask-based web application that uses **YOLO vehicle detection** and **Double Deep Q-Network (DDQN)** reinforcement learning to optimize traffic signal timing. This system analyzes traffic videos and provides intelligent recommendations for signal optimization.

## 📋 Features

### Core Functionality
- ✅ **YOLO Vehicle Detection** - Real-time vehicle counting using YOLOv8
- ✅ **DDQN Traffic Optimization** - AI-powered signal timing optimization
- ✅ **Multi-Lane Analysis** - Support for multiple traffic lanes
- ✅ **Video Processing** - Upload and process traffic videos
- ✅ **Real-time Visualization** - Charts and graphs of traffic metrics
- ✅ **User Authentication** - Secure login/registration system
- ✅ **Location Management** - Track multiple traffic locations
- ✅ **Analysis History** - Store and review past analyses

### Technical Features
- Pure vanilla JavaScript frontend (no jQuery)
- Inline CSS (no external stylesheets needed)
- RESTful API architecture
- SQLite database
- Asynchronous video processing
- Canvas-based chart rendering

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Folder Structure
```bash
mkdir -p templates static/charts static/videos uploads outputs
```

### 3. Place Template Files
Move all HTML files to the `templates/` folder:
- home.html
- login.html
- register.html
- signal_training.html
- signal_inventory.html
- analysis_result.html

### 4. Run Application
```bash
python app.py
```

Visit: `http://localhost:5000`

## 📖 Usage Guide

### 1. Register an Account
1. Navigate to http://localhost:5000/register
2. Fill in username, email, password
3. Click "Create Account"

### 2. Create a Location
1. Go to "Signal Training" page
2. Add location with name, address, and number of lanes
3. Click "Add Location"

### 3. Run Traffic Analysis
1. Select a location
2. Upload traffic video (MP4, AVI, MOV)
3. Click "Start Analysis"
4. Wait for processing (several minutes for long videos)

### 4. View Results
1. Click "View" on completed analysis
2. Review metrics and watch processed video
3. View traffic flow charts
4. Read AI recommendations

## 🔧 Configuration

### Key Settings in app.py

```python
# Video upload limit
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# YOLO model (change for speed/accuracy)
model = YOLO('yolov8n.pt')  # n=fastest, x=most accurate

# Signal timing
cooldown_frames = 30  # Minimum frames between changes
```

## 🎓 How It Works

### YOLO Vehicle Detection
Detects and counts vehicles (car, motorcycle, bus, truck, bicycle) in each frame.

### DDQN Traffic Optimization
- **State**: Vehicle counts per lane
- **Action**: Which lane gets green light
- **Reward**: Reduced wait time + balanced lanes

The AI learns optimal signal timing through experience replay and Q-learning.

## 📊 API Endpoints

### Authentication
- POST /api/auth/login
- POST /api/auth/register
- POST /api/auth/logout

### Dashboard
- GET /api/dashboard/stats
- GET /api/dashboard/activity

### Locations
- GET /api/locations
- POST /api/locations
- DELETE /api/locations/:id

### Analyses
- GET /api/analyses
- GET /api/analyses/:id
- POST /api/analyses

### Signals
- GET /api/signals
- PATCH /api/signals/:id

## 🐛 Troubleshooting

### YOLO Model Not Downloading
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Database Reset
```bash
rm traffic_monitor.db
python app.py
```

### Memory Issues
- Use smaller video files
- Use yolov8n (nano) model
- Reduce video resolution

## 🚀 Performance Tips

### GPU Acceleration
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Video Preprocessing
```bash
ffmpeg -i input.mp4 -vf scale=640:480 output.mp4
```

## 🔐 Production Deployment

1. Change SECRET_KEY in app.py
2. Use PostgreSQL instead of SQLite
3. Enable HTTPS with gunicorn
4. Add rate limiting
5. Implement CSRF protection

## 📁 Project Structure
```
trafficai-pro/
├── app.py                 # Main Flask app
├── requirements.txt       # Dependencies
├── templates/            # HTML files
├── static/
│   ├── charts/          # Generated charts
│   └── videos/          # Output videos
├── uploads/             # Input videos
└── traffic_monitor.db   # SQLite database
```

## 🤝 Key Technologies

- **Flask** - Web framework
- **YOLO (Ultralytics)** - Vehicle detection
- **PyTorch** - Deep learning
- **OpenCV** - Video processing
- **SQLAlchemy** - Database ORM
- **Matplotlib** - Chart generation

---

**Version**: 1.0  
**Last Updated**: October 2025
