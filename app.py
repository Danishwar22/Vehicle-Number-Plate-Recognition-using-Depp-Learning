# ========================================================================
# FLASK BACKEND - COMPLETE ANPR SYSTEM WITH AUTHENTICATION
# Student: DANISHWAR P (24MCA0029)
# Guide: Dr. ANGULAKSHMI M
# ========================================================================

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from functools import wraps
import os
import cv2
import numpy as np
from bson.objectid import ObjectId
import secrets

# YOLOv8 and OCR imports
from ultralytics import YOLO
import easyocr

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/anpr_database'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# Initialize MongoDB
mongo = PyMongo(app)

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load YOLOv8 model
MODEL_PATH = 'models/best.pt'
try:
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except:
    model = None
    print(f"⚠ Model not found at {MODEL_PATH}")

# Initialize EasyOCR
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("✓ EasyOCR initialized (GPU)")
except:
    reader = easyocr.Reader(['en'], gpu=False)
    print("✓ EasyOCR initialized (CPU)")

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def extract_text(image, bbox):
    """Extract text from license plate using EasyOCR"""
    try:
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        w = max(0, min(w, w_img - x))
        h = max(0, min(h, h_img - y))
        
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0 or 0 in roi.shape:
            return 'NO_TEXT'
        
        # Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # OCR
        result = reader.readtext(gray, detail=0)
        text = ''.join(result).strip().upper().replace(' ', '')
        
        return text if text else 'NO_TEXT'
    except Exception as e:
        print(f"OCR Error: {e}")
        return 'ERROR'

def process_image(image_path, user_id, conf_threshold=0.5):
    """Process single image and detect plates"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Failed to read image'}
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(img_rgb, conf=conf_threshold, verbose=False)
        
        detections = []
        img_draw = img.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                bbox = [x1, y1, x2-x1, y2-y1]
                plate_text = extract_text(img, bbox)
                
                # Draw on image
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_draw, f'{plate_text} {conf*100:.1f}%', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                detection = {
                    'plate_number': plate_text,
                    'confidence': round(conf, 4),
                    'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                    'timestamp': datetime.now()
                }
                detections.append(detection)
                
                # Save to detected_plates collection
                mongo.db.detected_plates.insert_one({
                    'plate_number': plate_text,
                    'confidence': round(conf, 4),
                    'user_id': user_id,
                    'detected_at': datetime.now(),
                    'source_image': os.path.basename(image_path)
                })
        
        # Save result image
        result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(image_path)}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, img_draw)
        
        return {
            'success': True,
            'detections': detections,
            'total_plates': len(detections),
            'result_image': result_filename
        }
    except Exception as e:
        return {'error': str(e)}

def process_video(video_path, user_id, conf_threshold=0.5):
    """Process video and detect plates"""
    try:
        cap = cv2.VideoCapture(video_path)
        all_detections = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 30th frame
            if frame_count % 30 == 0:
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        bbox = [x1, y1, x2-x1, y2-y1]
                        plate_text = extract_text(frame, bbox)
                        
                        detection = {
                            'plate_number': plate_text,
                            'confidence': round(conf, 4),
                            'frame': frame_count
                        }
                        all_detections.append(detection)
                        
                        # Save to database
                        mongo.db.detected_plates.insert_one({
                            'plate_number': plate_text,
                            'confidence': round(conf, 4),
                            'user_id': user_id,
                            'detected_at': datetime.now(),
                            'source_video': os.path.basename(video_path),
                            'frame_number': frame_count
                        })
                
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        # Get unique plates
        unique_plates = list(set([d['plate_number'] for d in all_detections if d['plate_number'] not in ['NO_TEXT', 'ERROR']]))
        
        return {
            'success': True,
            'total_frames': frame_count,
            'processed_frames': processed_frames,
            'total_detections': len(all_detections),
            'unique_plates': unique_plates,
            'detections': all_detections
        }
    except Exception as e:
        return {'error': str(e)}

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/')
def landing():
    """Landing page with User/Admin choice"""
    return render_template('landing.html')

@app.route('/user/login')
def user_login_page():
    """User login page"""
    return render_template('user_login.html')

@app.route('/admin/login')
def admin_login_page():
    """Admin login page"""
    return render_template('admin_login.html')

@app.route('/user/register')
def user_register_page():
    """User registration page"""
    return render_template('user_register.html')

@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if password != confirm_password:
        return jsonify({'error': 'Passwords do not match'}), 400
    
    # Check if user exists
    if mongo.db.users.find_one({'username': username}):
        return jsonify({'error': 'Username already exists'}), 400
    
    # Create user
    user = {
        'username': username,
        'password': generate_password_hash(password),
        'role': 'user',
        'created_at': datetime.now()
    }
    
    result = mongo.db.users.insert_one(user)
    
    return jsonify({
        'success': True,
        'message': 'Registration successful',
        'user_id': str(result.inserted_id)
    })

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user = mongo.db.users.find_one({'username': username})
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if user['role'] != role:
        return jsonify({'error': f'Access denied. {role.capitalize()} access required'}), 403
    
    # Create session
    session['user_id'] = str(user['_id'])
    session['username'] = user['username']
    session['role'] = user['role']
    session.permanent = True
    
    return jsonify({
        'success': True,
        'user': {
            'id': str(user['_id']),
            'username': user['username'],
            'role': user['role']
        }
    })

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# ==================== USER DASHBOARD ====================

@app.route('/user/dashboard')
@login_required
def user_dashboard():
    """User dashboard"""
    return render_template('user_dashboard.html')

@app.route('/user/check-number')
@login_required
def check_number_page():
    """Check number plate page"""
    return render_template('check_number.html')

@app.route('/user/upload-stolen')
@login_required
def upload_stolen_page():
    """Upload stolen vehicle page"""
    return render_template('upload_stolen.html')

@app.route('/user/check-stolen')
@login_required
def check_stolen_page():
    """Check if vehicle is stolen"""
    return render_template('check_stolen.html')

# ==================== USER API ROUTES ====================

@app.route('/api/check-number', methods=['POST'])
@login_required
def check_number():
    """Upload and detect number plate"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Check if image or video
    ext = filename.rsplit('.', 1)[1].lower()
    user_id = session['user_id']
    
    if ext in ['jpg', 'jpeg', 'png']:
        result = process_image(filepath, user_id)
    elif ext in ['mp4', 'avi', 'mov', 'mkv']:
        result = process_video(filepath, user_id)
    else:
        return jsonify({'error': 'Unsupported file format'}), 400
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/api/upload-stolen', methods=['POST'])
@login_required
def upload_stolen():
    """Report stolen vehicle"""
    data = request.json
    plate_number = data.get('plate_number', '').upper().replace(' ', '')
    owner_name = data.get('owner_name')
    contact = data.get('contact')
    vehicle_info = data.get('vehicle_info')
    
    if not plate_number:
        return jsonify({'error': 'Plate number required'}), 400
    
    # Check if already reported
    existing = mongo.db.stolen_vehicles.find_one({'plate_number': plate_number, 'status': 'stolen'})
    if existing:
        return jsonify({'error': 'Vehicle already reported as stolen'}), 400
    
    # Insert stolen vehicle
    stolen = {
        'plate_number': plate_number,
        'owner_name': owner_name,
        'contact': contact,
        'vehicle_info': vehicle_info,
        'reported_by': session['user_id'],
        'reported_at': datetime.now(),
        'status': 'stolen'
    }
    
    result = mongo.db.stolen_vehicles.insert_one(stolen)
    
    return jsonify({
        'success': True,
        'message': 'Vehicle reported as stolen',
        'record_id': str(result.inserted_id)
    })

@app.route('/api/check-stolen', methods=['POST'])
@login_required
def check_stolen():
    """Check if plate number is stolen"""
    data = request.json
    plate_number = data.get('plate_number', '').upper().replace(' ', '')
    
    if not plate_number:
        return jsonify({'error': 'Plate number required'}), 400
    
    # Search in stolen vehicles
    stolen = mongo.db.stolen_vehicles.find_one({
        'plate_number': plate_number,
        'status': 'stolen'
    })
    
    if stolen:
        # Check if this plate was detected
        detections = list(mongo.db.detected_plates.find({
            'plate_number': plate_number
        }).sort('detected_at', -1).limit(5))
        
        return jsonify({
            'success': True,
            'is_stolen': True,
            'stolen_info': {
                'plate_number': stolen['plate_number'],
                'owner_name': stolen.get('owner_name'),
                'contact': stolen.get('contact'),
                'vehicle_info': stolen.get('vehicle_info'),
                'reported_at': stolen['reported_at'].isoformat()
            },
            'recent_detections': [
                {
                    'detected_at': d['detected_at'].isoformat(),
                    'confidence': d['confidence'],
                    'source': d.get('source_image') or d.get('source_video')
                } for d in detections
            ]
        })
    
    return jsonify({
        'success': True,
        'is_stolen': False,
        'message': 'Vehicle not reported as stolen'
    })

# ==================== ADMIN DASHBOARD ====================

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    return render_template('admin_dashboard.html')

@app.route('/api/admin/statistics', methods=['GET'])
@admin_required
def admin_statistics():
    """Get admin statistics"""
    total_users = mongo.db.users.count_documents({'role': 'user'})
    total_detections = mongo.db.detected_plates.count_documents({})
    total_stolen = mongo.db.stolen_vehicles.count_documents({'status': 'stolen'})
    total_found = mongo.db.stolen_vehicles.count_documents({'status': 'found'})
    
    return jsonify({
        'success': True,
        'statistics': {
            'total_users': total_users,
            'total_detections': total_detections,
            'total_stolen': total_stolen,
            'total_found': total_found
        }
    })

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    """Get all registered users"""
    users = list(mongo.db.users.find({'role': 'user'}))
    
    for user in users:
        user['_id'] = str(user['_id'])
        user.pop('password', None)
        user['created_at'] = user.get('created_at', datetime.now()).isoformat()
    
    return jsonify({
        'success': True,
        'users': users
    })

@app.route('/api/admin/stolen-vehicles', methods=['GET'])
@admin_required
def get_stolen_vehicles():
    """Get all stolen vehicles"""
    stolen = list(mongo.db.stolen_vehicles.find({'status': 'stolen'}).sort('reported_at', -1))
    
    for s in stolen:
        s['_id'] = str(s['_id'])
        s['reported_at'] = s['reported_at'].isoformat()
    
    return jsonify({
        'success': True,
        'stolen_vehicles': stolen
    })

@app.route('/api/admin/found-vehicles', methods=['GET'])
@admin_required
def get_found_vehicles():
    """Get all found vehicles"""
    found = list(mongo.db.stolen_vehicles.find({'status': 'found'}).sort('found_at', -1))
    
    for f in found:
        f['_id'] = str(f['_id'])
        f['reported_at'] = f['reported_at'].isoformat()
        f['found_at'] = f.get('found_at', datetime.now()).isoformat()
    
    return jsonify({
        'success': True,
        'found_vehicles': found
    })

@app.route('/api/admin/mark-found/<vehicle_id>', methods=['POST'])
@admin_required
def mark_found(vehicle_id):
    """Mark stolen vehicle as found"""
    result = mongo.db.stolen_vehicles.update_one(
        {'_id': ObjectId(vehicle_id)},
        {'$set': {'status': 'found', 'found_at': datetime.now()}}
    )
    
    if result.modified_count > 0:
        return jsonify({'success': True, 'message': 'Vehicle marked as found'})
    
    return jsonify({'error': 'Vehicle not found'}), 404

@app.route('/api/admin/detections', methods=['GET'])
@admin_required
def get_all_detections():
    """Get all detections"""
    detections = list(mongo.db.detected_plates.find().sort('detected_at', -1).limit(100))
    
    for d in detections:
        d['_id'] = str(d['_id'])
        d['detected_at'] = d['detected_at'].isoformat()
    
    return jsonify({
        'success': True,
        'detections': detections
    })

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result images"""
    from flask import send_from_directory
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# Initialize admin user on first run
def init_admin():
    """Create default admin if not exists"""
    admin = mongo.db.users.find_one({'role': 'admin'})
    if not admin:
        admin_user = {
            'username': 'admin',
            'password': generate_password_hash('admin123'),
            'role': 'admin',
            'created_at': datetime.now()
        }
        mongo.db.users.insert_one(admin_user)
        print("✓ Default admin created (username: admin, password: admin123)")

if __name__ == '__main__':
    init_admin()
    print("\n" + "="*80)
    print("ANPR SYSTEM - BACKEND SERVER STARTED")
    print("="*80)
    print(f"Model: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"OCR: {'✓ Ready' if reader else '✗ Not Ready'}")
    print(f"Default Admin - Username: admin, Password: admin123")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)