from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta
import json
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
from functools import wraps
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic_monitor.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

db = SQLAlchemy(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/graphs', exist_ok=True)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    locations = db.relationship('Location', backref='user', lazy=True, cascade='all, delete-orphan')

class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200))
    city = db.Column(db.String(100))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    num_lanes = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    signals = db.relationship('Signal', backref='location', lazy=True, cascade='all, delete-orphan')
    analyses = db.relationship('Analysis', backref='location', lazy=True, cascade='all, delete-orphan')

class Signal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'), nullable=False)
    signal_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='active')
    efficiency_score = db.Column(db.Float, default=0.0)
    avg_wait_time = db.Column(db.Float, default=0.0)
    vehicles_processed = db.Column(db.Integer, default=0)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'), nullable=False)
    signal_id = db.Column(db.Integer, db.ForeignKey('signal.id'), nullable=True)
    output_video_path = db.Column(db.String(200))
    reward_graph_path = db.Column(db.String(200))
    count_graph_path = db.Column(db.String(200))
    metrics_data = db.Column(db.Text)
    status = db.Column(db.String(20), default='processing')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    signal = db.relationship('Signal', backref='analyses')

# AI Models
class DQN(nn.Module):
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
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_lanes - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.q_eval(state_tensor)).item()

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def learn(self):
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
        self.q_target.load_state_dict(self.q_eval.state_dict())

VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

def get_vehicle_count(model, frame):
    result = model(frame)[0]
    return sum(1 for box in result.boxes if model.names[int(box.cls[0])] in VEHICLE_CLASSES)

def read_lane_frames(cap_list):
    return [cap.read()[1] if cap.isOpened() else None for cap in cap_list]

def run_traffic_controller(n_lanes, video_paths, output_path):
    model = YOLO('yolo11l.pt')
    agent = DDQNAgent(n_lanes)
    
    caps = [cv2.VideoCapture(p) for p in video_paths]
    width, height = int(caps[0].get(3)), int(caps[0].get(4))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    target_height = min(int(cap.get(4)) for cap in caps)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * n_lanes, target_height))
    
    frame_count = 0
    rewards = []
    car_counts = []
    
    while True:
        frames = read_lane_frames(caps)
        if any(f is None for f in frames):
            break
            
        frames = [cv2.resize(frame, (width, target_height)) for frame in frames]
        counts = [get_vehicle_count(model, f) for f in frames]
        action = agent.choose_action(counts)
        reward = -counts[action]
        next_counts = counts[:]
        next_counts[action] = max(0, next_counts[action] - random.randint(2, 5))
        agent.store_transition(counts, action, reward, next_counts)
        agent.learn()
        
        if frame_count % 20 == 0:
            agent.update_target()
            
        duration = min(10, max(3, counts[action]))
        combined_frame = np.hstack(frames)
        
        for i, count in enumerate(counts):
            signal = "GREEN" if i == action else "RED"
            label = f"Lane {i+1}: {signal} ({duration:.1f}s)"
            color = (0, 255, 0) if i == action else (0, 0, 255)
            cv2.putText(combined_frame, label, (i * width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
        out.write(combined_frame)
        frame_count += 1
        rewards.append(reward)
        car_counts.append(counts)
        
    for cap in caps:
        cap.release()
    out.release()
    
    return rewards, car_counts

def generate_graphs(rewards, car_counts, analysis_id):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, linewidth=2, color='#667eea')
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Negative Reward", fontsize=12)
    plt.title("Traffic Optimization Performance", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    reward_path = f'static/graphs/reward_{analysis_id}.png'
    plt.savefig(reward_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    car_counts_np = np.array(car_counts)
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
    for i in range(car_counts_np.shape[1]):
        plt.plot(car_counts_np[:, i], label=f"Lane {i+1}", linewidth=2, color=colors[i % len(colors)])
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Vehicle Count", fontsize=12)
    plt.title("Vehicle Distribution Across Lanes", fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    count_path = f'static/graphs/count_{analysis_id}.png'
    plt.savefig(count_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return reward_path, count_path

def process_analysis_async(analysis_id, location_id, video_files):
    with app.app_context():
        analysis = Analysis.query.get(analysis_id)
        location = Location.query.get(location_id)
        
        try:
            output_filename = f'output_{analysis.id}.mp4'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            rewards, car_counts = run_traffic_controller(location.num_lanes, video_files, output_path)
            
            reward_graph, count_graph = generate_graphs(rewards, car_counts, analysis.id)
            
            total_vehicles = sum(sum(counts) for counts in car_counts)
            avg_vehicles = total_vehicles / len(car_counts) if car_counts else 0
            efficiency = max(0, min(100, ((abs(rewards[0]) - abs(rewards[-1])) / abs(rewards[0]) * 100))) if rewards else 0
            
            analysis.output_video_path = output_path
            analysis.reward_graph_path = reward_graph
            analysis.count_graph_path = count_graph
            analysis.metrics_data = json.dumps({
                'rewards': rewards,
                'car_counts': car_counts,
                'total_vehicles': total_vehicles,
                'avg_vehicles': avg_vehicles,
                'efficiency': efficiency
            })
            analysis.status = 'completed'
            
            signal = Signal.query.filter_by(location_id=location_id).first()
            if not signal:
                signal = Signal(
                    location_id=location_id,
                    signal_name=f"{location.name} - Signal",
                    status='active'
                )
                db.session.add(signal)
            
            signal.efficiency_score = efficiency
            signal.vehicles_processed += total_vehicles
            signal.last_active = datetime.utcnow()
            analysis.signal_id = signal.id
            
            db.session.commit()
            
        except Exception as e:
            analysis.status = 'failed'
            db.session.commit()
            print(f"Analysis failed: {str(e)}")

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
            
        hashed_password = generate_password_hash(password)
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'success': True})
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/signal-training')
@login_required
def signal_training():
    return render_template('signal_training.html')

@app.route('/signal-inventory')
@login_required
def signal_inventory():
    return render_template('signal_inventory.html')

@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if analysis.location.user_id != session['user_id']:
        return redirect(url_for('home'))
    
    return render_template('analysis_result.html', analysis=analysis)

# API Endpoints
@app.route('/api/locations', methods=['GET'])
@login_required
def get_locations():
    locations = Location.query.filter_by(user_id=session['user_id']).all()
    return jsonify([{
        'id': loc.id,
        'name': loc.name,
        'address': loc.address,
        'city': loc.city,
        'num_lanes': loc.num_lanes,
        'latitude': loc.latitude,
        'longitude': loc.longitude,
        'created_at': loc.created_at.isoformat()
    } for loc in locations])

@app.route('/api/location/add', methods=['POST'])
@login_required
def add_location():
    data = request.get_json()
    
    location = Location(
        name=data.get('name'),
        address=data.get('address'),
        city=data.get('city'),
        latitude=float(data.get('latitude')) if data.get('latitude') else None,
        longitude=float(data.get('longitude')) if data.get('longitude') else None,
        num_lanes=int(data.get('num_lanes')),
        user_id=session['user_id']
    )
    db.session.add(location)
    db.session.commit()
    
    return jsonify({'success': True, 'location_id': location.id})

@app.route('/api/location/<int:location_id>/analysis', methods=['POST'])
@login_required
def start_analysis(location_id):
    location = Location.query.get_or_404(location_id)
    
    if location.user_id != session['user_id']:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    video_files = []
    for i in range(location.num_lanes):
        file = request.files.get(f'lane_{i}')
        if file:
            filename = secure_filename(f'{location_id}_lane_{i}_{datetime.now().timestamp()}.mp4')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            video_files.append(filepath)
    
    if len(video_files) != location.num_lanes:
        return jsonify({'success': False, 'error': 'Missing videos'}), 400
    
    analysis = Analysis(location_id=location_id, status='processing')
    db.session.add(analysis)
    db.session.commit()
    
    thread = threading.Thread(target=process_analysis_async, args=(analysis.id, location_id, video_files))
    thread.start()
    
    return jsonify({
        'success': True,
        'analysis_id': analysis.id
    })

@app.route('/api/dashboard-stats')
@login_required
def dashboard_stats():
    active_signals = Signal.query.join(Location).filter(
        Location.user_id == session['user_id'],
        Signal.status == 'active'
    ).all()
    
    recent_analyses = Analysis.query.join(Location).filter(
        Location.user_id == session['user_id'],
        Analysis.status == 'completed'
    ).order_by(Analysis.created_at.desc()).limit(5).all()
    
    total_locations = Location.query.filter_by(user_id=session['user_id']).count()
    total_signals = Signal.query.join(Location).filter(Location.user_id == session['user_id']).count()
    total_analyses = Analysis.query.join(Location).filter(Location.user_id == session['user_id']).count()
    
    analyses = Analysis.query.join(Location).filter(
        Location.user_id == session['user_id'],
        Analysis.status == 'completed'
    ).order_by(Analysis.created_at.desc()).limit(7).all()
    
    dates = []
    efficiencies = []
    vehicle_counts = []
    
    for analysis in analyses:
        dates.append(analysis.created_at.strftime('%m/%d'))
        metrics = json.loads(analysis.metrics_data) if analysis.metrics_data else {}
        efficiencies.append(metrics.get('efficiency', 0))
        vehicle_counts.append(metrics.get('total_vehicles', 0))
    
    return jsonify({
        'active_signals': [{
            'id': s.id,
            'name': s.signal_name,
            'location': s.location.name,
            'status': s.status,
            'efficiency': s.efficiency_score,
            'vehicles': s.vehicles_processed,
            'last_active': s.last_active.isoformat()
        } for s in active_signals],
        'recent_analyses': [{
            'id': a.id,
            'location': a.location.name,
            'status': a.status,
            'created_at': a.created_at.isoformat()
        } for a in recent_analyses],
        'stats': {
            'total_locations': total_locations,
            'total_signals': total_signals,
            'total_analyses': total_analyses
        },
        'chart_data': {
            'dates': dates[::-1],
            'efficiencies': efficiencies[::-1],
            'vehicle_counts': vehicle_counts[::-1]
        }
    })

@app.route('/api/signals')
@login_required
def get_signals():
    signals = Signal.query.join(Location).filter(
        Location.user_id == session['user_id']
    ).all()
    
    return jsonify([{
        'id': s.id,
        'name': s.signal_name,
        'location': s.location.name,
        'location_id': s.location_id,
        'status': s.status,
        'efficiency_score': s.efficiency_score,
        'vehicles_processed': s.vehicles_processed,
        'last_active': s.last_active.isoformat(),
        'created_at': s.created_at.isoformat()
    } for s in signals])

@app.route('/api/signal/<int:signal_id>/toggle', methods=['POST'])
@login_required
def toggle_signal(signal_id):
    signal = Signal.query.get_or_404(signal_id)
    
    if signal.location.user_id != session['user_id']:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    if signal.status == 'active':
        signal.status = 'inactive'
    else:
        signal.status = 'active'
        signal.last_active = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({'success': True, 'status': signal.status})

@app.route('/api/analysis/<int:analysis_id>/status')
@login_required
def analysis_status(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if analysis.location.user_id != session['user_id']:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    return jsonify({
        'status': analysis.status,
        'completed': analysis.status == 'completed'
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)