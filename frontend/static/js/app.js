// Elements
const videoEl = document.getElementById('webcam');
const overlayEl = document.getElementById('camera-overlay');
const hiddenCanvas = document.getElementById('hidden-canvas');
const ctx = hiddenCanvas.getContext('2d');

const connTitle = document.getElementById('conn-title');
const connDetail = document.getElementById('conn-detail');
const connCard = document.getElementById('conn-status-card');
const sessionIdEl = document.getElementById('session-id');
const camStatus = document.getElementById('cam-status');
const micStatus = document.getElementById('mic-status');

const pillRppg = document.getElementById('pill-rppg');
const pillAcoustic = document.getElementById('pill-acoustic');
const pillSync = document.getElementById('pill-sync');
const resultBox = document.getElementById('result-box');

// State
let ws = null;
let mediaStream = null;
let audioContext = null;
let scriptProcessor = null;
let sourceNode = null;
let isConnected = false;
let animationFrameId = null;

// Config
const TARGET_FPS = 15; // Limit browser CPU usage
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
let lastFrameTime = 0;

// Initialize
async function init() {
    try {
        updateConnStatus("CONNECTING", "Opening WebSocket to backend...", "#5a9abf");
        
        // 1. Setup WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/session`;
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            isConnected = true;
            updateConnStatus("CONNECTED", "Live biometric stream active", "#00ff88");
            startMediaCapture();
        };
        
        ws.onclose = () => {
            isConnected = false;
            updateConnStatus("DISCONNECTED", "Backend unreachable. Retrying...", "#ff4a4a");
            stopMediaCapture();
            setTimeout(init, 3000);
        };
        
        ws.onerror = (err) => {
            console.error("WebSocket error", err);
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleBiometricFrame(data);
            } catch (e) {
                console.error("Failed to parse message", e);
            }
        };
        
    } catch (e) {
        console.error("Init failed", e);
        updateConnStatus("DISCONNECTED", e.message, "#ff4a4a");
    }
}

// Media Capture
async function startMediaCapture() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser API unsupported or Insecure Context (needs HTTPS/localhost)");
        }

        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, frameRate: 30, facingMode: "user" },
            audio: true
        });
        
        // Video
        videoEl.srcObject = mediaStream;
        videoEl.play();
        overlayEl.classList.add("hidden");
        camStatus.innerHTML = "<span style='color:#00ff88'>✅ Active</span>";
        
        // Audio
        setupAudioPipeline(mediaStream);
        micStatus.innerHTML = "<span style='color:#00ff88'>✅ Active</span>";
        
        // Start transmit loops
        hiddenCanvas.width = 640;
        hiddenCanvas.height = 480;
        animationFrameId = requestAnimationFrame(processVideoFrame);

    } catch (err) {
        console.error("Media error:", err);
        overlayEl.classList.remove("hidden");
        document.getElementById('overlay-text').innerText = "🚫 " + err.message;
        camStatus.innerHTML = "<span style='color:#ff4a4a'>❌ Denied/Error</span>";
        micStatus.innerHTML = "<span style='color:#ff4a4a'>❌ Denied/Error</span>";
    }
}

function stopMediaCapture() {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    if (scriptProcessor) scriptProcessor.disconnect();
    if (sourceNode) sourceNode.disconnect();
    if (audioContext) audioContext.close();
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
}

// Audio Pipeline (PCM extraction)
function setupAudioPipeline(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioContext.createMediaStreamSource(stream);
    
    // 2048 buffer size is a good balance between latency and overhead
    scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
    
    scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
        if (!isConnected || ws.readyState !== WebSocket.OPEN) return;
        
        const inputBuffer = audioProcessingEvent.inputBuffer;
        const inputData = inputBuffer.getChannelData(0); // Float32Array
        
        // Convert to standard array for JSON serialization
        const samples = Array.from(inputData);
        
        const msg = {
            type: "audio_chunk",
            timestamp: Date.now() / 1000.0,
            sample_rate: audioContext.sampleRate,
            data: samples
        };
        
        ws.send(JSON.stringify(msg));
    };
    
    sourceNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
}

// Video Pipeline
function processVideoFrame(timestamp) {
    if (!isConnected || ws.readyState !== WebSocket.OPEN) return;
    
    if (timestamp - lastFrameTime >= FRAME_INTERVAL_MS) {
        lastFrameTime = timestamp;
        
        if (videoEl.readyState === videoEl.HAVE_ENOUGH_DATA) {
            ctx.drawImage(videoEl, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
            // Quality 0.6 to save bandwidth
            const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.6);
            // Strip 'data:image/jpeg;base64,'
            const base64Data = dataUrl.split(',')[1];
            
            const msg = {
                type: "video_frame",
                timestamp: Date.now() / 1000.0,
                data: base64Data
            };
            ws.send(JSON.stringify(msg));
        }
    }
    
    animationFrameId = requestAnimationFrame(processVideoFrame);
}

// UI Updates
function updateConnStatus(state, detail, color) {
    connTitle.innerText = state;
    connTitle.style.color = color;
    connDetail.innerText = detail;
    connCard.style.borderColor = color + "44";
}

function handleBiometricFrame(frame) {
    if (frame.session_id) {
        sessionIdEl.innerText = frame.session_id.substring(0, 8);
    }
    
    updatePill(pillRppg, "Liveness", frame.rppg_liveness);
    updatePill(pillAcoustic, "Voice Authenticity", frame.audio_trust);
    updatePill(pillSync, "Audio-Visual Sync", frame.sync_score);
    
    renderResultBox(frame.status_flag, frame.overall_trust);
}

function updatePill(el, label, score) {
    el.className = "signal-pill"; // reset
    if (score === null || score === undefined) {
        el.classList.add("signal-na");
        el.innerHTML = `<strong>${label}</strong> &nbsp; ⟳ Calc...`;
    } else {
        // Sub-scores from backend are [0, 1]
        const safeScore = Math.max(0, Math.min(1, score));
        const pct = Math.round(safeScore * 100);
        if (safeScore >= 0.70) {
            el.classList.add("signal-pass");
            el.innerHTML = `<strong>${label}</strong> &nbsp; ✓ ${pct}%`;
        } else {
            el.classList.add("signal-fail");
            el.innerHTML = `<strong>${label}</strong> &nbsp; ✗ ${pct}%`;
        }
    }
}

function renderResultBox(status, trust) {
    resultBox.classList.remove("hidden");
    resultBox.className = "result-box";
    
    // overall_trust from backend is already [0, 100]. Do not multiply by 100.
    const rawTrust = trust !== null ? trust : 0;
    const safeTrust = Math.max(0, Math.min(100, rawTrust));
    const scoreVal = safeTrust.toFixed(1);
    
    if (status === "nominal") {
        resultBox.classList.add("result-pass");
        resultBox.innerHTML = `
            <div class="result-title" style="color:var(--green)">🟢 HUMAN / CONSISTENT SIGNALS</div>
            <div class="score" style="color:var(--green)">${scoreVal}<span class="score-max"> / 100</span></div>
            <div class="result-reason" style="color:var(--green)">Analysis confirms active liveness signals.</div>
        `;
    } else if (status === "calibrating") {
        resultBox.classList.add("result-pass");
        resultBox.style.background = "rgba(255, 204, 0, 0.05)";
        resultBox.style.borderColor = "rgba(255, 204, 0, 0.2)";
        // Show provisional score during calibration clearly marked as calculating
        resultBox.innerHTML = `
            <div class="result-title" style="color:var(--yellow)">🟡 CALIBRATING</div>
            <div class="score" style="color:var(--yellow)">${scoreVal}<span class="score-max"> / 100 (Provisional)</span></div>
            <div class="result-reason" style="color:var(--yellow)">Collecting biometric samples. Please hold still and speak.</div>
        `;
    } else {
        resultBox.classList.add("result-fail");
        resultBox.innerHTML = `
            <div class="result-title" style="color:var(--red)">🔴 SUSPICIOUS ACTIVITY</div>
            <div class="score" style="color:var(--red)">${scoreVal}<span class="score-max"> / 100</span></div>
            <div class="result-reason" style="color:var(--red)">Signals indicate possible synthesis or spoofing.</div>
        `;
    }
}

// Start
init();
