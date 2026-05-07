/**
 * Thai Coin Detection - Frontend App
 * ===================================
 * Camera management, detection API calls, and result rendering
 */

// ==========================================
// State
// ==========================================
let cameraStream = null;
let isAutoDetecting = false;
let autoDetectInterval = null;
let detectionLog = [];
let lastFrameTime = 0;
let frameCount = 0;
let currentFps = 0;

const video = document.getElementById('camera-feed');
const canvas = document.getElementById('detection-canvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('camera-placeholder');

// Bbox colors per class
// Merged dataset: 4 classes (1-baht, 2-baht, 5-baht, 10-baht)
const BBOX_COLORS = {
    '1 Baht':  { stroke: '#b0bec5', fill: 'rgba(176,190,197,0.15)', text: '#e0e0e0' },
    '2 Baht':  { stroke: '#ffd54f', fill: 'rgba(255,213,79,0.15)',  text: '#fff9c4' },
    '5 Baht':  { stroke: '#90a4ae', fill: 'rgba(144,164,174,0.15)', text: '#cfd8dc' },
    '10 Baht': { stroke: '#ff8a65', fill: 'rgba(255,138,101,0.15)', text: '#ffccbc' },
};

// ==========================================
// Camera
// ==========================================
async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera(deviceId = null) {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 960 },
                facingMode: deviceId ? undefined : 'environment'
            },
            audio: false
        };
        if (deviceId) constraints.video.deviceId = { exact: deviceId };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = cameraStream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        placeholder.classList.add('hidden');
        document.getElementById('btn-camera-text').textContent = 'ปิดกล้อง';
        document.getElementById('btn-detect').disabled = false;
        document.getElementById('btn-auto').disabled = false;
        
        updateStatus(true);
        enumerateCameras();
    } catch (err) {
        console.error('Camera error:', err);
        alert('ไม่สามารถเปิดกล้องได้: ' + err.message);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    video.srcObject = null;
    placeholder.classList.remove('hidden');
    document.getElementById('btn-camera-text').textContent = 'เปิดกล้อง';
    document.getElementById('btn-detect').disabled = true;
    document.getElementById('btn-auto').disabled = true;
    
    if (isAutoDetecting) toggleAutoDetect();
    clearCanvas();
    updateStatus(false);
}

async function enumerateCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        const select = document.getElementById('camera-select');
        select.innerHTML = '<option value="">เลือกกล้อง</option>';
        cameras.forEach((cam, i) => {
            const opt = document.createElement('option');
            opt.value = cam.deviceId;
            opt.textContent = cam.label || `กล้อง ${i + 1}`;
            select.appendChild(opt);
        });
    } catch (e) { /* ignore */ }
}

function switchCamera() {
    const select = document.getElementById('camera-select');
    if (select.value) {
        stopCamera();
        startCamera(select.value);
    }
}

// ==========================================
// Detection
// ==========================================
async function captureAndDetect() {
    if (!cameraStream) return;

    // Show scanning animation
    showScanLine();

    // Capture frame
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(video, 0, 0);
    const imageData = captureCanvas.toDataURL('image/jpeg', 0.85);

    try {
        const res = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await res.json();
        
        if (data.success) {
            renderDetections(data);
            updateSummary(data);
            updateBreakdown(data);
            addToLog(data);
        } else {
            console.error('Detection failed:', data.error);
        }
    } catch (err) {
        console.error('API error:', err);
    }

    removeScanLine();
    updateFps();
}

function toggleAutoDetect() {
    isAutoDetecting = !isAutoDetecting;
    const btn = document.getElementById('btn-auto');
    const text = document.getElementById('btn-auto-text');

    if (isAutoDetecting) {
        btn.classList.add('active');
        text.textContent = 'Auto: On';
        document.getElementById('fps-counter').classList.remove('hidden');
        autoDetectLoop();
    } else {
        btn.classList.remove('active');
        text.textContent = 'Auto: Off';
        document.getElementById('fps-counter').classList.add('hidden');
        if (autoDetectInterval) {
            clearTimeout(autoDetectInterval);
            autoDetectInterval = null;
        }
    }
}

async function autoDetectLoop() {
    if (!isAutoDetecting || !cameraStream) return;
    await captureAndDetect();
    // ~3-5 FPS for detection (adjust based on performance)
    autoDetectInterval = setTimeout(autoDetectLoop, 300);
}

// ==========================================
// Rendering
// ==========================================
function renderDetections(data) {
    clearCanvas();
    if (!data.detections || data.detections.length === 0) return;

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    data.detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
        const sx2 = x2 * scaleX, sy2 = y2 * scaleY;
        const w = sx2 - sx1, h = sy2 - sy1;

        const colors = BBOX_COLORS[det.class_name] || { stroke: '#fff', fill: 'rgba(255,255,255,0.1)', text: '#fff' };

        // Fill
        ctx.fillStyle = colors.fill;
        ctx.fillRect(sx1, sy1, w, h);

        // Border
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([]);
        ctx.strokeRect(sx1, sy1, w, h);

        // Corner accents
        const cornerLen = Math.min(w, h) * 0.25;
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = colors.stroke;
        // Top-left
        ctx.beginPath(); ctx.moveTo(sx1, sy1 + cornerLen); ctx.lineTo(sx1, sy1); ctx.lineTo(sx1 + cornerLen, sy1); ctx.stroke();
        // Top-right
        ctx.beginPath(); ctx.moveTo(sx2 - cornerLen, sy1); ctx.lineTo(sx2, sy1); ctx.lineTo(sx2, sy1 + cornerLen); ctx.stroke();
        // Bottom-left
        ctx.beginPath(); ctx.moveTo(sx1, sy2 - cornerLen); ctx.lineTo(sx1, sy2); ctx.lineTo(sx1 + cornerLen, sy2); ctx.stroke();
        // Bottom-right
        ctx.beginPath(); ctx.moveTo(sx2 - cornerLen, sy2); ctx.lineTo(sx2, sy2); ctx.lineTo(sx2, sy2 - cornerLen); ctx.stroke();

        // Label
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 14px Inter, sans-serif';
        const tm = ctx.measureText(label);
        const lw = tm.width + 12;
        const lh = 22;

        // Label background
        ctx.fillStyle = colors.stroke;
        ctx.beginPath();
        const lr = 4;
        ctx.roundRect(sx1, sy1 - lh - 2, lw, lh, [lr, lr, 0, 0]);
        ctx.fill();

        // Label text
        ctx.fillStyle = '#0a0e1a';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, sx1 + 6, sy1 - lh / 2 - 2);
    });

    // Show total overlay
    const overlay = document.getElementById('total-overlay');
    overlay.classList.remove('hidden');
    document.getElementById('total-value').textContent = `${data.total_value} ฿`;
    document.getElementById('total-count').textContent = `${data.total_coins} เหรียญ`;
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ==========================================
// UI Updates
// ==========================================
function updateSummary(data) {
    const totalEl = document.getElementById('summary-total');
    const coinsEl = document.getElementById('summary-coins');
    
    animateNumber(totalEl, parseInt(totalEl.textContent) || 0, data.total_value);
    animateNumber(coinsEl, parseInt(coinsEl.textContent) || 0, data.total_coins);
}

function animateNumber(el, from, to) {
    const duration = 400;
    const start = performance.now();
    
    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(from + (to - from) * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function updateBreakdown(data) {
    const container = document.getElementById('coin-breakdown');
    
    if (!data.coin_counts || Object.keys(data.coin_counts).length === 0) {
        container.innerHTML = '<div class="empty-state"><p>ไม่พบเหรียญ</p></div>';
        return;
    }

    const order = ['10 Baht', '5 Baht', '2 Baht', '1 Baht'];
    let html = '';

    order.forEach(name => {
        const info = data.coin_counts[name];
        if (!info) return;
        
        const valueNum = parseInt(name);
        html += `
            <div class="coin-row">
                <div class="coin-icon coin-${valueNum}">฿${valueNum}</div>
                <div class="coin-info">
                    <div class="coin-name">${name === '10 Baht' ? 'เหรียญ 10 บาท' : name === '5 Baht' ? 'เหรียญ 5 บาท' : name === '2 Baht' ? 'เหรียญ 2 บาท' : 'เหรียญ 1 บาท'}</div>
                    <div class="coin-count-label">${info.count} เหรียญ</div>
                </div>
                <div class="coin-subtotal">${info.subtotal} ฿</div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function addToLog(data) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('th-TH');
    
    detectionLog.unshift({
        time: timeStr,
        value: data.total_value,
        coins: data.total_coins
    });

    // Keep last 20
    if (detectionLog.length > 20) detectionLog.pop();

    const container = document.getElementById('detection-log');
    container.innerHTML = detectionLog.map(entry => `
        <div class="log-entry">
            <span class="log-time">${entry.time}</span>
            <span class="log-coins">${entry.coins} เหรียญ</span>
            <span class="log-value">${entry.value} ฿</span>
        </div>
    `).join('');
}

function updateStatus(online) {
    const badge = document.getElementById('badge-status');
    badge.textContent = online ? 'Online' : 'Offline';
    badge.className = `badge ${online ? 'badge-online' : 'badge-offline'}`;
}

// ==========================================
// Effects
// ==========================================
function showScanLine() {
    if (document.querySelector('.scanning-line')) return;
    const line = document.createElement('div');
    line.className = 'scanning-line';
    document.querySelector('.camera-wrapper').appendChild(line);
}

function removeScanLine() {
    const line = document.querySelector('.scanning-line');
    if (line) line.remove();
}

function updateFps() {
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
        currentFps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
        document.getElementById('fps-counter').textContent = `FPS: ${currentFps}`;
    }
}

// ==========================================
// Init
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    // Check API status
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            console.log('System status:', data);
            if (data.demo_mode) {
                console.log('🎮 Running in Demo Mode');
            }
        })
        .catch(err => console.error('Status check failed:', err));
});
