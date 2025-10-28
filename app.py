"""
TrafficAI Pro - Flask Backend
Integrates YOLO vehicle detection with DDQN traffic signal optimization
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import os
from datetime import datetime
import json
import threading
import cv2
import torch
import numpy as np
import random
from collections import deque
from ultralytics import YOLO
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Flask Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic_monitor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

db = SQLAlchemy(app)

# Ensure folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               os.path.join(app.config['STATIC_FOLDER'], 'charts'),
               os.path.join(app.config['STATIC_FOLDER'], 'videos')]:
    os.makedirs(folder, exist_ok=True)

# ============================================================================
# YOLO + DDQN TRAFFIC MONITORING CODE (PRESERVED FROM ORIGINAL)
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network for traffic signal optimization"""
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        return self.fc(x)


class DDQNAgent:
    """Double Deep Q-Network Agent for adaptive traffic control"""
    def __init__(self, n_lanes):
        self.n_lanes = n_lanes
        self.q_eval = DQN(n_lanes)
        self.q_target = DQN(n_lanes)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=0.001)

    def choose_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_lanes - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.q_eval(state_tensor)).item()

    def store_transition(self, s, a, r, s_):
        """Store experience in replay memory"""
        self.memory.append((s, a, r, s_))

    def learn(self):
        """Update Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_ = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)

        q_values = self.q_eval(s).gather(1, a)
        next_actions = self.q_eval(s_).argmax(1).unsqueeze(1)
        q_targets = self.q_target(s_).gather(1, next_actions)
        target = r + self.gamma * q_targets

        loss = nn.MSELoss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        """Sync target network with evaluation network"""
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def save_model(self, path):
        """Save trained model"""
        torch.save(self.q_eval.state_dict(), path)

    def load_model(self, path):
        """Load pre-trained model"""
        try:
            self.q_eval.load_state_dict(torch.load(path))
            self.q_target.load_state_dict(self.q_eval.state_dict())
            return True
        except FileNotFoundError:
            return False


# Vehicle classes recognized by YOLO
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']


def get_vehicle_count(model, frame):
    """Count vehicles in a frame using YOLO"""
    result = model(frame)[0]
    return sum(1 for box in result.boxes if model.names[int(box.cls[0])] in VEHICLE_CLASSES)


def read_lane_frames(cap_list):
    """Read frames from multiple video captures"""
    return [cap.read()[1] if cap.isOpened() else None for cap in cap_list]


def run_traffic_controller(n_lanes, video_paths, output_path, model_path="traffic_agent.pth", 
                          progress_callback=None):
    """
    Main traffic controller with YOLO detection and DDQN optimization
    
    Args:
        n_lanes: Number of traffic lanes
        video_paths: List of video file paths for each lane
        output_path: Output video path
        model_path: Path to saved DDQN model
        progress_callback: Optional callback function for progress updates
        
    Returns:
        dict: Analysis results including rewards, counts, and metrics
    """
    print(f"🚀 Starting traffic analysis for {n_lanes} lanes...")
    
    # Load YOLO model for vehicle detection
    model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed
    agent = DDQNAgent(n_lanes)
    
    # Try to load pre-trained model
    if agent.load_model(model_path):
        print(f"💡 Loaded previous knowledge from {model_path}")
    else:
        print("⚠️ No previous model found – starting fresh.")

    # Load all video sources (lanes)
    caps = [cv2.VideoCapture(p) for p in video_paths]
    
    if not all(cap.isOpened() for cap in caps):
        raise ValueError("❌ Could not open one or more video files")

    # Get video dimensions and properties
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    target_height = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)

    # Setup output video
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width * n_lanes, target_height)
    )

    frame_count = 0
    rewards, car_counts = [], []
    lane_metrics = {i: {'total_vehicles': 0, 'green_time': 0} for i in range(n_lanes)}

    cooldown_frames = 30  # Minimum frames between signal changes
    current_action = 0
    frame_since_action = cooldown_frames
    prev_counts = [0] * n_lanes

    print("🚦 Running adaptive traffic controller...")

    while True:
        frames = read_lane_frames(caps)
        if any(f is None for f in frames):
            break

        # Resize frames to uniform height
        frames = [cv2.resize(frame, (width, target_height)) for frame in frames]

        # Count vehicles in each lane
        counts = [get_vehicle_count(model, f) for f in frames]

        # Decide action (which lane gets green light)
        if frame_since_action >= cooldown_frames:
            action = agent.choose_action(counts)
            current_action = action
            frame_since_action = 0
        else:
            action = current_action
            frame_since_action += 1

        # Calculate reward (reduce total wait time and balance lanes)
        total_prev = sum(prev_counts)
        total_curr = sum(counts)
        reward = (total_prev - total_curr) - 0.1 * np.std(counts)
        prev_counts = counts[:]

        # Simulate next state (vehicles passing through green lane)
        next_counts = counts[:]
        next_counts[action] = max(0, next_counts[action] - random.randint(2, 5))

        # Store transition and learn
        agent.store_transition(counts, action, reward, next_counts)
        agent.learn()
        
        if frame_count % 20 == 0:
            agent.update_target()

        # Calculate signal duration based on traffic
        duration = max(3, min(10, sum(counts) // n_lanes))

        # Combine frames horizontally
        combined_frame = np.hstack(frames)

        # Draw signal status and vehicle counts
        for i, count in enumerate(counts):
            signal = "GREEN" if i == action else "RED"
            label = f"Lane {i+1}: {signal} ({duration:.1f}s)"
            color = (0, 255, 0) if i == action else (0, 0, 255)
            
            # Signal status
            cv2.putText(
                combined_frame, label,
                (i * width + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            
            # Vehicle count
            cv2.putText(
                combined_frame, f"Vehicles: {count}",
                (i * width + 10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Update lane metrics
            lane_metrics[i]['total_vehicles'] += count
            if i == action:
                lane_metrics[i]['green_time'] += 1

        # Write frame to output
        out.write(combined_frame)
        frame_count += 1

        # Store metrics
        rewards.append(reward)
        car_counts.append(counts)

        # Progress callback
        if progress_callback and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            progress_callback(progress, frame_count, total_frames)

    # Cleanup
    for cap in caps:
        cap.release()
    out.release()

    # Save trained agent
    agent.save_model(model_path)

    print(f"\n✅ Final traffic video saved at: {output_path}")
    print(f"📊 Average Reward: {round(np.mean(rewards), 3)}")
    print(f"🎯 Total Frames Processed: {frame_count}")

    # Calculate final statistics
    total_vehicles = sum(sum(counts) for counts in car_counts)
    avg_wait_time = calculate_avg_wait_time(car_counts, n_lanes)
    efficiency = calculate_efficiency(lane_metrics, frame_count)

    results = {
        'rewards': rewards,
        'car_counts': car_counts,
        'lane_metrics': lane_metrics,
        'total_frames': frame_count,
        'total_vehicles': total_vehicles,
        'avg_wait_time': avg_wait_time,
        'efficiency': efficiency,
        'throughput': total_vehicles / (frame_count / fps) if frame_count > 0 else 0
    }

    return results


def calculate_avg_wait_time(car_counts, n_lanes):
    """Calculate average wait time based on vehicle counts"""
    if not car_counts:
        return 0
    total_wait = sum(sum(counts) for counts in car_counts)
    return round(total_wait / (len(car_counts) * n_lanes), 2)


def calculate_efficiency(lane_metrics, total_frames):
    """Calculate traffic signal efficiency"""
    if total_frames == 0:
        return 0
    
    total_green_time = sum(metrics['green_time'] for metrics in lane_metrics.values())
    n_lanes = len(lane_metrics)
    optimal_green_time = total_frames / n_lanes
    
    efficiency = min(100, (total_green_time / (optimal_green_time * n_lanes)) * 100)
    return round(efficiency, 2)


def generate_charts(rewards, car_counts, analysis_id):
    """Generate analysis charts"""
    chart_dir = os.path.join(app.config['STATIC_FOLDER'], 'charts')
    
    # Reward chart
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.title("Traffic Optimization Reward Over Time")
    plt.grid(True)
    reward_path = os.path.join(chart_dir, f'reward_{analysis_id}.png')
    plt.savefig(reward_path)
    plt.close()

    # Car count chart
    plt.figure(figsize=(10, 6))
    car_counts_np = np.array(car_counts)
    for i in range(car_counts_np.shape[1]):
        plt.plot(car_counts_np[:, i], label=f"Lane {i+1}")
    plt.xlabel("Frame")
    plt.ylabel("Vehicle Count")
    plt.title("Vehicle Count Over Time per Lane")
    plt.legend()
    plt.grid(True)
    count_path = os.path.join(chart_dir, f'counts_{analysis_id}.png')
    plt.savefig(count_path)
    plt.close()

    return {
        'reward_chart': f'/static/charts/reward_{analysis_id}.png',
        'count_chart': f'/static/charts/counts_{analysis_id}.png'
    }


# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(db.Model):
    """User account model"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    locations = db.relationship('Location', backref='user', lazy=True, cascade='all, delete-orphan')


class Location(db.Model):
    """Traffic location model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    num_lanes = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    signals = db.relationship('Signal', backref='location', lazy=True, cascade='all, delete-orphan')
    analyses = db.relationship('Analysis', backref='location', lazy=True, cascade='all, delete-orphan')


class Signal(db.Model):
    """Traffic signal model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'), nullable=False)
    status = db.Column(db.String(20), default='active')  # active, inactive
    phases = db.Column(db.Integer, default=4)
    cycle_time = db.Column(db.Integer, default=120)  # seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Analysis(db.Model):
    """Traffic analysis results model"""
    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'), nullable=False)
    video_path = db.Column(db.String(500))
    output_video = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    total_vehicles = db.Column(db.Integer)
    avg_wait_time = db.Column(db.Float)
    throughput = db.Column(db.Float)
    efficiency = db.Column(db.Float)
    duration = db.Column(db.String(50))
    peak_hour = db.Column(db.String(50))
    results_json = db.Column(db.Text)  # Store detailed results
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)


# ============================================================================
# AUTHENTICATION HELPERS
# ============================================================================

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# TEMPLATE ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')


@app.route('/register')
def register():
    """Registration page"""
    return render_template('register.html')


@app.route('/signal-training')
def signal_training():
    """Signal training page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('signal_training.html')


@app.route('/signal-inventory')
def signal_inventory():
    """Signal inventory page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('signal_inventory.html')


@app.route('/analysis/<int:analysis_id>')
def analysis_result(analysis_id):
    """Analysis results page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('analysis_result.html')


# ============================================================================
# AUTHENTICATION API
# ============================================================================

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """User login"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'redirect': '/'
        }), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """User registration"""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 400

    hashed_password = generate_password_hash(password)
    user = User(username=username, email=email, password=hashed_password)
    
    db.session.add(user)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Registration successful'
    }), 201


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """User logout"""
    session.clear()
    return jsonify({'success': True}), 200


# ============================================================================
# DASHBOARD API
# ============================================================================

@app.route('/api/dashboard/stats')
@login_required
def api_dashboard_stats():
    """Get dashboard statistics"""
    user_id = session['user_id']
    
    locations = Location.query.filter_by(user_id=user_id).count()
    signals = Signal.query.join(Location).filter(Location.user_id == user_id).count()
    analyses = Analysis.query.join(Location).filter(Location.user_id == user_id).count()

    return jsonify({
        'locations': locations,
        'signals': signals,
        'analyses': analyses
    })


@app.route('/api/dashboard/activity')
@login_required
def api_dashboard_activity():
    """Get recent activity"""
    user_id = session['user_id']
    
    recent_analyses = Analysis.query.join(Location).filter(
        Location.user_id == user_id
    ).order_by(Analysis.created_at.desc()).limit(5).all()

    activities = []
    for analysis in recent_analyses:
        activities.append({
            'type': 'analysis',
            'title': f'Analysis completed for {analysis.location.name}',
            'timestamp': analysis.created_at.isoformat()
        })

    return jsonify(activities)


# ============================================================================
# LOCATION API
# ============================================================================

@app.route('/api/locations', methods=['GET'])
@login_required
def api_get_locations():
    """Get all locations"""
    user_id = session['user_id']
    locations = Location.query.filter_by(user_id=user_id).all()
    
    return jsonify([{
        'id': loc.id,
        'name': loc.name,
        'address': loc.address,
        'num_lanes': loc.num_lanes,
        'created_at': loc.created_at.isoformat()
    } for loc in locations])


@app.route('/api/locations', methods=['POST'])
@login_required
def api_create_location():
    """Create new location"""
    data = request.get_json()
    user_id = session['user_id']
    
    location = Location(
        name=data['name'],
        address=data.get('address'),
        num_lanes=data['num_lanes'],
        user_id=user_id
    )
    
    db.session.add(location)
    db.session.commit()

    return jsonify({
        'success': True,
        'id': location.id,
        'message': 'Location created'
    }), 201


@app.route('/api/locations/<int:location_id>', methods=['DELETE'])
@login_required
def api_delete_location(location_id):
    """Delete location"""
    user_id = session['user_id']
    location = Location.query.filter_by(id=location_id, user_id=user_id).first()
    
    if not location:
        return jsonify({'error': 'Location not found'}), 404
    
    db.session.delete(location)
    db.session.commit()

    return jsonify({'success': True}), 200


# ============================================================================
# ANALYSIS API
# ============================================================================

@app.route('/api/analyses', methods=['GET'])
@login_required
def api_get_analyses():
    """Get all analyses"""
    user_id = session['user_id']
    limit = request.args.get('limit', 10, type=int)
    
    analyses = Analysis.query.join(Location).filter(
        Location.user_id == user_id
    ).order_by(Analysis.created_at.desc()).limit(limit).all()

    return jsonify([{
        'id': a.id,
        'location_name': a.location.name,
        'status': a.status,
        'created_at': a.created_at.isoformat()
    } for a in analyses])


@app.route('/api/analyses/<int:analysis_id>', methods=['GET'])
@login_required
def api_get_analysis(analysis_id):
    """Get specific analysis"""
    user_id = session['user_id']
    analysis = Analysis.query.join(Location).filter(
        Analysis.id == analysis_id,
        Location.user_id == user_id
    ).first()

    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    result = {
        'id': analysis.id,
        'location_name': analysis.location.name,
        'status': analysis.status,
        'total_vehicles': analysis.total_vehicles,
        'avg_wait_time': analysis.avg_wait_time,
        'throughput': analysis.throughput,
        'efficiency': analysis.efficiency,
        'output_video': analysis.output_video,
        'duration': analysis.duration,
        'peak_hour': analysis.peak_hour,
        'created_at': analysis.created_at.isoformat()
    }

    if analysis.results_json:
        try:
            detailed_results = json.loads(analysis.results_json)
            result.update(detailed_results)
        except:
            pass

    return jsonify(result)


@app.route('/api/analyses', methods=['POST'])
@login_required
def api_create_analysis():
    """Start new traffic analysis"""
    user_id = session['user_id']
    location_id = request.form.get('location_id')
    
    location = Location.query.filter_by(id=location_id, user_id=user_id).first()
    if not location:
        return jsonify({'error': 'Location not found'}), 404

    # Handle video upload or URL
    video_paths = []
    
    if 'video' in request.files:
        # Single video file
        video = request.files['video']
        if video.filename:
            filename = secure_filename(video.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{timestamp}_{filename}')
            video.save(save_path)
            # Duplicate for each lane (in real scenario, you'd have separate videos)
            video_paths = [save_path] * location.num_lanes
    
    elif 'video_url' in request.form:
        # Video URL (would need to download first)
        video_url = request.form.get('video_url')
        # For now, use URL directly (requires modification to support URLs in run_traffic_controller)
        video_paths = [video_url] * location.num_lanes

    if not video_paths:
        return jsonify({'error': 'No video provided'}), 400

    # Create analysis record
    analysis = Analysis(
        location_id=location_id,
        video_path=json.dumps(video_paths),
        status='processing'
    )
    
    db.session.add(analysis)
    db.session.commit()

    # Start processing in background thread
    thread = threading.Thread(
        target=process_analysis_async,
        args=(analysis.id, location.num_lanes, video_paths)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'id': analysis.id,
        'message': 'Analysis started'
    }), 201


def process_analysis_async(analysis_id, n_lanes, video_paths):
    """Process traffic analysis asynchronously"""
    with app.app_context():
        analysis = Analysis.query.get(analysis_id)
        
        try:
            # Output path
            output_filename = f'analysis_{analysis_id}.mp4'
            output_path = os.path.join(app.config['STATIC_FOLDER'], 'videos', output_filename)
            
            # Run traffic controller
            results = run_traffic_controller(
                n_lanes=n_lanes,
                video_paths=video_paths,
                output_path=output_path
            )
            
            # Generate charts
            charts = generate_charts(
                results['rewards'],
                results['car_counts'],
                analysis_id
            )
            
            # Update analysis record
            analysis.status = 'completed'
            analysis.output_video = f'/static/videos/{output_filename}'
            analysis.total_vehicles = results['total_vehicles']
            analysis.avg_wait_time = results['avg_wait_time']
            analysis.throughput = results['throughput']
            analysis.efficiency = results['efficiency']
            analysis.duration = f"{results['total_frames'] / 30:.1f}s"  # Assuming 30fps
            analysis.completed_at = datetime.utcnow()
            
            # Store detailed results
            detailed_results = {
                'charts': charts,
                'lane_metrics': results['lane_metrics']
            }
            analysis.results_json = json.dumps(detailed_results)
            
            db.session.commit()
            
            print(f"✅ Analysis {analysis_id} completed successfully")
            
        except Exception as e:
            print(f"❌ Analysis {analysis_id} failed: {str(e)}")
            analysis.status = 'failed'
            db.session.commit()


# ============================================================================
# SIGNAL API
# ============================================================================

@app.route('/api/signals', methods=['GET'])
@login_required
def api_get_signals():
    """Get all signals"""
    user_id = session['user_id']
    
    signals = Signal.query.join(Location).filter(
        Location.user_id == user_id
    ).all()

    return jsonify([{
        'id': sig.id,
        'name': sig.name or f'Signal {sig.id}',
        'location_name': sig.location.name,
        'status': sig.status,
        'phases': sig.phases,
        'cycle_time': sig.cycle_time
    } for sig in signals])


@app.route('/api/signals/<int:signal_id>', methods=['PATCH'])
@login_required
def api_update_signal(signal_id):
    """Update signal"""
    user_id = session['user_id']
    data = request.get_json()
    
    signal = Signal.query.join(Location).filter(
        Signal.id == signal_id,
        Location.user_id == user_id
    ).first()

    if not signal:
        return jsonify({'error': 'Signal not found'}), 404

    if 'status' in data:
        signal.status = data['status']
    
    signal.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({'success': True}), 200


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")


if __name__ == '__main__':
    init_db()
    print("🚀 TrafficAI Pro Backend Started")
    print("📡 Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
