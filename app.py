# -*- coding: utf-8 -*-
"""
Thai Coin Detection - Flask Backend
====================================
ตรวจจับเหรียญไทย (1, 2, 5, 10 บาท) ด้วย YOLOv8
รองรับการตรวจจับหลายเหรียญพร้อมกันและนับมูลค่ารวม

Usage:
    python app.py              # รันปกติ (ต้องมี model)
    python app.py --demo       # รันโหมด demo (ไม่ต้องมี model)
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os
import sys
import time
import random

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# ==========================================
# Configuration
# ==========================================
# Unified classes from merged Roboflow datasets
# coin-thai v9 + thai-coins-model v4
COIN_CLASSES = {
    0: {'name': '1 Baht', 'value': 1, 'color': [192, 192, 192]},   # Silver
    1: {'name': '2 Baht', 'value': 2, 'color': [0, 215, 255]},     # Gold (brass)
    2: {'name': '5 Baht', 'value': 5, 'color': [192, 192, 192]},   # Silver
    3: {'name': '10 Baht', 'value': 10, 'color': [0, 165, 255]},   # Bi-metallic
}

MODEL_PATH = os.path.join('models', 'thai_coins_best.pt')
CONFIDENCE_THRESHOLD = 0.5
DEMO_MODE = '--demo' in sys.argv

# ==========================================
# Model Loading
# ==========================================
model = None

def load_model():
    """Load YOLOv8 model."""
    global model
    if DEMO_MODE:
        print("[DEMO] Running in DEMO mode (no real model)")
        return True
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at {MODEL_PATH}")
        print("   Run 'python train.py' first, or use '--demo' flag")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print(f"[OK] Model loaded: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False


def generate_demo_detections(width=640, height=480):
    """Generate fake detections for demo mode."""
    num_coins = random.randint(3, 8)
    detections = []
    
    for _ in range(num_coins):
        class_id = random.choice([0, 1, 2, 3])
        coin = COIN_CLASSES[class_id]
        
        # Random position (avoid edges)
        size = random.randint(40, 80)
        x1 = random.randint(20, width - size - 20)
        y1 = random.randint(20, height - size - 20)
        x2 = x1 + size
        y2 = y1 + size
        
        confidence = round(random.uniform(0.75, 0.99), 2)
        
        detections.append({
            'class_id': class_id,
            'class_name': coin['name'],
            'value': coin['value'],
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2],
            'color': coin['color']
        })
    
    return detections


def detect_coins(image_data):
    """
    Detect coins in an image.
    
    Args:
        image_data: Base64 encoded image or numpy array
    
    Returns:
        dict with detections and summary
    """
    if DEMO_MODE:
        detections = generate_demo_detections()
        total_value = sum(d['value'] for d in detections)
        total_coins = len(detections)
        
        # Count by type
        coin_counts = {}
        for d in detections:
            name = d['class_name']
            if name not in coin_counts:
                coin_counts[name] = {'count': 0, 'subtotal': 0}
            coin_counts[name]['count'] += 1
            coin_counts[name]['subtotal'] += d['value']
        
        return {
            'success': True,
            'detections': detections,
            'total_value': total_value,
            'total_coins': total_coins,
            'coin_counts': coin_counts,
            'demo_mode': True
        }
    
    # Decode base64 image
    try:
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data
    except Exception as e:
        return {'success': False, 'error': f'Image decode error: {str(e)}'}
    
    if img is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Run YOLO detection
    try:
        results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                coin = COIN_CLASSES.get(class_id, {
                    'name': f'Unknown({class_id})',
                    'value': 0,
                    'color': [128, 128, 128]
                })
                
                detections.append({
                    'class_id': class_id,
                    'class_name': coin['name'],
                    'value': coin['value'],
                    'confidence': round(confidence, 2),
                    'bbox': [x1, y1, x2, y2],
                    'color': coin['color']
                })
        
        # Calculate totals
        total_value = sum(d['value'] for d in detections)
        total_coins = len(detections)
        
        # Count by type
        coin_counts = {}
        for d in detections:
            name = d['class_name']
            if name not in coin_counts:
                coin_counts[name] = {'count': 0, 'subtotal': 0}
            coin_counts[name]['count'] += 1
            coin_counts[name]['subtotal'] += d['value']
        
        return {
            'success': True,
            'detections': detections,
            'total_value': total_value,
            'total_coins': total_coins,
            'coin_counts': coin_counts,
            'demo_mode': False
        }
    
    except Exception as e:
        return {'success': False, 'error': f'Detection error: {str(e)}'}


# ==========================================
# Routes
# ==========================================
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', demo_mode=DEMO_MODE)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    API endpoint for coin detection.
    Accepts base64 encoded image in JSON body.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        result = detect_coins(data['image'])
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status')
def api_status():
    """Check system status."""
    return jsonify({
        'status': 'online',
        'demo_mode': DEMO_MODE,
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'classes': {str(k): v['name'] for k, v in COIN_CLASSES.items()},
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  Thai Coin Detection System")
    print("  Detect Thai coins with YOLOv8")
    print("=" * 60)
    print()
    
    if load_model():
        print()
        print(f"[SERVER] Starting at http://localhost:5000")
        print(f"[DEMO] Mode: {'ON' if DEMO_MODE else 'OFF'}")
        print()
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        print()
        print("[TIP] Use 'python app.py --demo' to test without model")
        print("[TIP] Use 'python train.py' to train model")
        sys.exit(1)
