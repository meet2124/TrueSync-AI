/**
 * frontend/static/js/app.js
 * TrueSync AI — Verification Dashboard
 *
 * WebSocket protocol is UNCHANGED from the backend contract:
 *   Inbound: { type:"video_frame", data:"<base64 JPEG>" }
 *             { type:"audio_chunk", data:[float,...], sample_rate:16000 }
 *   Outbound: BiometricFrame JSON (see backend/core/schemas.py)
 *
 * All displayed values come directly from backend messages.
 * Nothing is fabricated or mocked in this JS.
 *
 * Phase 2A additions — device-agnostic media capture:
 *   - Device enumeration after permission granted (enumerateDevices)
 *   - Camera selector (hidden unless ≥2 cameras detected)
 *   - Camera switch WITHOUT restarting WebSocket or audio pipeline
 *   - OverconstrainedError fallback (retries without facingMode hint)
 *   - Track-ended detection (USB camera disconnection mid-session)
 *   - Conditional mirror flip (front vs rear camera)
 *   - iOS AudioContext suspend/resume handling
 */

'use strict';

/* ─── Element references ──────────────────────────────────────────── */
const videoEl          = document.getElementById('webcam');
const hiddenCanvas     = document.getElementById('hidden-canvas');
const ctx              = hiddenCanvas.getContext('2d');
const cameraOverlay    = document.getElementById('camera-overlay');
const overlayTextEl    = document.getElementById('overlay-text');
const scanLine         = document.getElementById('scan-line');
const faceGuide        = document.getElementById('face-guide');

const connBanner       = document.getElementById('conn-banner');
const connDot          = document.getElementById('conn-dot');
const connLabel        = document.getElementById('conn-label');

const engineBadge      = document.getElementById('engine-badge');
const engineDot        = document.getElementById('engine-dot');
const engineModeLabel  = document.getElementById('engine-mode-label');
const sessionIdEl      = document.getElementById('session-id');

const chipCam          = document.getElementById('chip-cam');
const camStatusText    = document.getElementById('cam-status-text');
const chipMic          = document.getElementById('chip-mic');
const micStatusText    = document.getElementById('mic-status-text');

const bpmValueEl       = document.getElementById('bpm-value');
const rppgCanvas       = document.getElementById('rppg-canvas');
const waveformIdle     = document.getElementById('waveform-idle');
const rppgCtx          = rppgCanvas.getContext('2d');

const gaugeArc         = document.getElementById('gauge-arc');
const trustScoreEl     = document.getElementById('trust-score');

const sigRppg          = document.getElementById('sig-rppg');
const scoreRppg        = document.getElementById('score-rppg');
const barRppg          = document.getElementById('bar-rppg');

const sigAcoustic      = document.getElementById('sig-acoustic');
const scoreAcoustic    = document.getElementById('score-acoustic');
const barAcoustic      = document.getElementById('bar-acoustic');

const sigSync          = document.getElementById('sig-sync');
const scoreSync        = document.getElementById('score-sync');
const barSync          = document.getElementById('bar-sync');

const decisionCard     = document.getElementById('decision-card');
const decisionIcon     = document.getElementById('decision-icon');
const decisionTitle    = document.getElementById('decision-title');
const decisionDetail   = document.getElementById('decision-detail');

// Phase 2A — device selector
const cameraSelectWrap = document.getElementById('camera-select-wrap');
const cameraSelect     = document.getElementById('camera-select');

/* ─── State ───────────────────────────────────────────────────────── */
let ws               = null;
/**
 * mediaStream — original combined stream from getUserMedia (audio + initial video).
 * Audio pipeline stays connected to this stream's audio tracks for the full session.
 */
let mediaStream      = null;
/**
 * videoStream — current video-only stream. Equal to mediaStream initially.
 * After a camera switch it points to a new video-only stream while
 * mediaStream (and its audio tracks) remain untouched.
 */
let videoStream      = null;
let audioContext     = null;
let scriptProcessor  = null;
let sourceNode       = null;
let isConnected      = false;
let animFrameId      = null;
let reconnectTimer   = null;
let reconnectDelay   = 2000;   // ms — backs off on repeated failures
const MAX_RECONNECT  = 16000;

/* Video capture config */
const TARGET_FPS       = 15;
const FRAME_INTERVAL   = 1000 / TARGET_FPS;
let lastFrameTime      = 0;

/* rPPG waveform ring buffer */
const WAVEFORM_POINTS  = 150;
let waveformBuffer     = new Array(WAVEFORM_POINTS).fill(null);
let waveformHasData    = false;

/* Gauge arc geometry — arc length at 100% */
const GAUGE_ARC_LEN    = 283;  // must match SVG stroke-dasharray

/* ─── Connection banner helpers ───────────────────────────────────── */
const CONNECTION_STATES = {
    connecting:   { cls: 'state-connecting',   text: 'Connecting to verification engine…' },
    connected:    { cls: 'state-connected',     text: 'Connected — live biometric stream active' },
    processing:   { cls: 'state-processing',    text: 'Processing — receiving biometric data' },
    disconnected: { cls: 'state-disconnected',  text: 'Disconnected — reconnecting…' },
    error:        { cls: 'state-error',          text: 'Connection error' },
};

function setConnectionState(state, customText) {
    const def = CONNECTION_STATES[state] || CONNECTION_STATES.connecting;
    const allCls = Object.values(CONNECTION_STATES).map(d => d.cls);
    connBanner.classList.remove(...allCls);
    connBanner.classList.add(def.cls);
    connLabel.textContent = customText || def.text;
}

/* ─── WebSocket init ──────────────────────────────────────────────── */
function init() {
    if (ws && ws.readyState < WebSocket.CLOSING) return;

    setConnectionState('connecting');

    try {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${window.location.host}/ws/session`);
    } catch (e) {
        setConnectionState('error', `WebSocket creation failed: ${e.message}`);
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        isConnected    = true;
        reconnectDelay = 2000;  // reset backoff
        setConnectionState('connected');
        startMediaCapture();
    };

    ws.onclose = (ev) => {
        isConnected = false;
        stopMediaCapture();
        const reason = ev.reason ? ` (${ev.reason})` : '';
        setConnectionState('disconnected', `Disconnected${reason} — reconnecting in ${reconnectDelay / 1000}s`);
        scheduleReconnect();
    };

    ws.onerror = () => {
        // onerror fires before onclose — log only; onclose handles reconnect
        setConnectionState('error', 'WebSocket error — check backend is running');
    };

    ws.onmessage = (ev) => {
        try {
            const frame = JSON.parse(ev.data);
            setConnectionState('processing');
            handleBiometricFrame(frame);
        } catch (e) {
            console.warn('[TrueSync] Failed to parse backend message:', e);
        }
    };
}

function scheduleReconnect() {
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => {
        reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT);
        init();
    }, reconnectDelay);
}

/* ─── Media capture ───────────────────────────────────────────────── */
async function startMediaCapture() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraState('error', 'Camera API unavailable (needs HTTPS or localhost)');
        setMicState('error', '—');
        return;
    }

    // ── Attempt 1: preferred constraints including facingMode hint ─────
    let acquired = false;
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, frameRate: 30, facingMode: 'user' },
            audio: true,
        });
        acquired = true;
    } catch (err) {
        if (err.name === 'OverconstrainedError') {
            // ── Attempt 2: relax constraints — drop facingMode hint ────
            console.warn('[TrueSync] OverconstrainedError — retrying without facingMode constraint');
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, frameRate: 30 },
                    audio: true,
                });
                acquired = true;
            } catch (fallbackErr) {
                handleMediaError(fallbackErr);
                return;
            }
        } else {
            handleMediaError(err);
            return;
        }
    }

    if (!acquired || !mediaStream) return;

    // ── Video element ─────────────────────────────────────────────────
    videoStream = mediaStream;       // initially the same combined stream
    videoEl.srcObject = mediaStream;
    await videoEl.play().catch(() => {});   // autoplay may need user gesture
    cameraOverlay.classList.add('hidden');
    scanLine.classList.add('active');
    faceGuide.classList.add('visible');
    setCameraState('active', 'CAM');
    updateMirror();

    // ── Audio pipeline ────────────────────────────────────────────────
    setupAudioPipeline(mediaStream);
    setMicState('active', 'MIC');

    // ── Video transmit loop ───────────────────────────────────────────
    hiddenCanvas.width  = 640;
    hiddenCanvas.height = 480;
    animFrameId = requestAnimationFrame(videoLoop);

    // ── Track-ended listener: detect USB camera disconnection ─────────
    addTrackEndedListener();

    // ── Device enumeration: labels only available after permission ─────
    await enumerateVideoDevices();
}

/** Unified error handler for getUserMedia failures. */
function handleMediaError(err) {
    console.error('[TrueSync] Media error:', err);
    let msg = err.message || 'Unknown media error';
    if (err.name === 'NotAllowedError')       msg = 'Permission denied — please allow camera & microphone';
    if (err.name === 'NotFoundError')         msg = 'No camera or microphone found on this device';
    if (err.name === 'NotReadableError')      msg = 'Camera is already in use by another application';
    if (err.name === 'OverconstrainedError')  msg = 'Camera constraints could not be satisfied — try a different camera';
    overlayTextEl.textContent = msg;
    cameraOverlay.classList.remove('hidden');
    scanLine.classList.remove('active');
    faceGuide.classList.remove('visible');
    setCameraState('error', 'CAM');
    setMicState('error', 'MIC');
}

function stopMediaCapture() {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (scriptProcessor) { try { scriptProcessor.disconnect(); } catch (_) {} scriptProcessor = null; }
    if (sourceNode)      { try { sourceNode.disconnect(); }      catch (_) {} sourceNode = null;      }
    if (audioContext)    { try { audioContext.close(); }          catch (_) {} audioContext = null;    }

    // Stop the switched-to video stream if it is separate from the original
    if (videoStream && videoStream !== mediaStream) {
        videoStream.getVideoTracks().forEach(t => t.stop());
    }
    videoStream = null;

    // Stop all tracks in the original stream (includes audio)
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }

    scanLine.classList.remove('active');
    faceGuide.classList.remove('visible');
}

function setCameraState(state, text) {
    chipCam.className = 'media-chip';
    if (state === 'active') chipCam.classList.add('chip-active');
    if (state === 'error')  chipCam.classList.add('chip-error');
    camStatusText.textContent = text;
}

function setMicState(state, text) {
    chipMic.className = 'media-chip';
    if (state === 'active') chipMic.classList.add('chip-active');
    if (state === 'error')  chipMic.classList.add('chip-error');
    micStatusText.textContent = text;
}

/* ─── Mirror: front cameras → mirror; rear cameras → no mirror ────── */
/**
 * Front-facing cameras (selfie mode) are mirrored so the view feels natural.
 * Rear cameras (environment) are NOT mirrored — that would flip text/signs.
 * Desktop cameras usually have no facingMode → default to mirror.
 */
function updateMirror() {
    const activeStream = videoStream || mediaStream;
    if (!activeStream) return;
    const track = activeStream.getVideoTracks()[0];
    if (!track) return;
    const { facingMode } = track.getSettings();
    const shouldMirror = !facingMode || facingMode === 'user';
    videoEl.style.transform = shouldMirror ? 'scaleX(-1)' : 'none';
}

/* ─── Track-ended: detect camera disconnection mid-session ─────────── */
function addTrackEndedListener() {
    const activeStream = videoStream || mediaStream;
    if (!activeStream) return;
    const track = activeStream.getVideoTracks()[0];
    if (!track) return;

    track.addEventListener('ended', async () => {
        console.warn('[TrueSync] Video track ended — camera disconnected?');
        overlayTextEl.textContent = 'Camera disconnected. Reconnect device or select another camera.';
        cameraOverlay.classList.remove('hidden');
        scanLine.classList.remove('active');
        faceGuide.classList.remove('visible');
        setCameraState('error', 'CAM');
        // Re-enumerate so selector reflects available cameras after reconnect
        await enumerateVideoDevices();
    });
}

/* ─── Device enumeration ─────────────────────────────────────────── */
/**
 * Enumerate video input devices and populate the camera selector.
 *
 * MUST be called AFTER getUserMedia() has been granted — the browser only
 * returns non-empty device labels once permission exists.
 *
 * The selector is rendered only when 2 or more cameras are found.
 */
async function enumerateVideoDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

    try {
        const devices    = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter(d => d.kind === 'videoinput');

        if (videoInputs.length < 2) {
            // Only one camera — selector adds no value
            cameraSelectWrap.classList.add('hidden');
            return;
        }

        // Populate <select> with real device labels
        cameraSelect.innerHTML = '';
        videoInputs.forEach((device, i) => {
            const opt = document.createElement('option');
            opt.value       = device.deviceId;
            opt.textContent = device.label || `Camera ${i + 1}`;
            cameraSelect.appendChild(opt);
        });

        // Pre-select whichever camera is currently active
        const activeStream = videoStream || mediaStream;
        if (activeStream) {
            const activeTrack = activeStream.getVideoTracks()[0];
            if (activeTrack) {
                const { deviceId } = activeTrack.getSettings();
                if (deviceId) cameraSelect.value = deviceId;
            }
        }

        cameraSelectWrap.classList.remove('hidden');
    } catch (e) {
        console.warn('[TrueSync] enumerateDevices() failed:', e);
    }
}

/* ─── Camera switch (video only — audio pipeline stays alive) ──────── */
cameraSelect.addEventListener('change', async () => {
    const deviceId = cameraSelect.value;
    if (!deviceId) return;
    await switchCamera(deviceId);
});

/**
 * Switch to a different camera by deviceId.
 *
 * Only the video track is replaced.  The audio pipeline (AudioContext,
 * ScriptProcessor, WebSocket) and the biometric engine session remain
 * completely undisturbed — the backend sees a seamless frame stream.
 */
async function switchCamera(deviceId) {
    // ── Stop old video track only ─────────────────────────────────────
    const prevStream = videoStream || mediaStream;
    if (prevStream) {
        prevStream.getVideoTracks().forEach(t => t.stop());
    }
    videoStream    = null;
    videoEl.srcObject = null;

    // ── Acquire new video-only stream ─────────────────────────────────
    try {
        const newVideoStream = await navigator.mediaDevices.getUserMedia({
            // audio: false — leave the existing audio pipeline untouched
            video: { deviceId: { exact: deviceId }, width: 640, height: 480, frameRate: 30 },
        });

        videoStream       = newVideoStream;
        videoEl.srcObject = newVideoStream;
        await videoEl.play().catch(() => {});

        cameraOverlay.classList.add('hidden');
        scanLine.classList.add('active');
        faceGuide.classList.add('visible');
        setCameraState('active', 'CAM');

        updateMirror();
        addTrackEndedListener();

        console.info('[TrueSync] Camera switched → deviceId:', deviceId);
    } catch (err) {
        console.error('[TrueSync] Camera switch failed:', err);
        let msg = `Camera switch failed: ${err.message}`;
        if (err.name === 'NotAllowedError')  msg = 'Permission denied for selected camera';
        if (err.name === 'NotReadableError') msg = 'Selected camera is in use by another application';
        overlayTextEl.textContent = msg;
        cameraOverlay.classList.remove('hidden');
        setCameraState('error', 'CAM');
    }
}

/* ─── Audio pipeline ─────────────────────────────────────────────── */
function setupAudioPipeline(stream) {
    audioContext    = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode      = audioContext.createMediaStreamSource(stream);
    scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);

    // iOS Safari creates AudioContext in 'suspended' state when there has been
    // no user gesture.  Resume on first interaction so audio chunks flow.
    if (audioContext.state === 'suspended') {
        const resumeCtx = () => {
            if (audioContext) audioContext.resume();
            document.removeEventListener('click',      resumeCtx);
            document.removeEventListener('touchstart', resumeCtx);
        };
        document.addEventListener('click',      resumeCtx, { once: true });
        document.addEventListener('touchstart', resumeCtx, { once: true });
    }

    scriptProcessor.onaudioprocess = (event) => {
        if (!isConnected || !ws || ws.readyState !== WebSocket.OPEN) return;

        const pcm     = event.inputBuffer.getChannelData(0); // Float32Array
        const samples = Array.from(pcm);

        ws.send(JSON.stringify({
            type:        'audio_chunk',
            timestamp:   Date.now() / 1000.0,
            sample_rate: audioContext.sampleRate,
            data:        samples,
        }));
    };

    sourceNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
}

/* ─── Video transmit loop ────────────────────────────────────────── */
function videoLoop(ts) {
    // videoEl.srcObject is always the current camera (updated on switch)
    if (!isConnected || !ws || ws.readyState !== WebSocket.OPEN) {
        animFrameId = requestAnimationFrame(videoLoop);
        return;
    }

    if (ts - lastFrameTime >= FRAME_INTERVAL) {
        lastFrameTime = ts;

        if (videoEl.readyState === videoEl.HAVE_ENOUGH_DATA) {
            ctx.drawImage(videoEl, 0, 0, 640, 480);
            const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.6);
            const b64     = dataUrl.split(',')[1];

            ws.send(JSON.stringify({
                type:      'video_frame',
                timestamp: Date.now() / 1000.0,
                data:      b64,
            }));
        }
    }

    animFrameId = requestAnimationFrame(videoLoop);
}

/* ─── BiometricFrame handler ─────────────────────────────────────── */
/**
 * Receives BiometricFrame fields:
 *   session_id, engine_mode, bpm, rppg_waveform, rppg_liveness,
 *   audio_trust, sync_score, overall_trust, active_rois, status_flag
 *
 * Sub-scores (rppg_liveness, audio_trust, sync_score) are [0..1].
 * overall_trust is [0..100].
 */
function handleBiometricFrame(frame) {
    // Session & engine info
    if (frame.session_id) {
        sessionIdEl.textContent = frame.session_id.substring(0, 8).toUpperCase();
    }

    const mode = frame.engine_mode || '';
    engineModeLabel.textContent = mode.toUpperCase() || '—';
    engineDot.className = 'badge-dot';
    if (mode === 'production') engineDot.classList.add('dot-production');
    else if (mode === 'mock')  engineDot.classList.add('dot-mock');

    // BPM
    updateBpm(frame.bpm);

    // rPPG waveform
    if (Array.isArray(frame.rppg_waveform) && frame.rppg_waveform.length > 0) {
        updateWaveformBuffer(frame.rppg_waveform);
    }
    drawWaveform();

    // Signal cards — sub-scores [0..1] from backend
    updateSignalCard(sigRppg,     scoreRppg,     barRppg,     frame.rppg_liveness);
    updateSignalCard(sigAcoustic, scoreAcoustic, barAcoustic, frame.audio_trust);
    updateSignalCard(sigSync,     scoreSync,     barSync,     frame.sync_score);

    // Trust gauge — overall_trust is [0..100]
    updateTrustGauge(frame.overall_trust);

    // Verification decision
    updateDecision(frame.status_flag, frame.overall_trust);
}

/* ─── BPM display ────────────────────────────────────────────────── */
function updateBpm(bpm) {
    if (bpm === null || bpm === undefined) {
        bpmValueEl.textContent = '—';
        bpmValueEl.style.color = '';
    } else {
        const v = Math.round(clamp(bpm, 0, 300));
        bpmValueEl.textContent = v;
        // Colour-code by physiological plausibility
        if (v >= 45 && v <= 120)      bpmValueEl.style.color = 'var(--clr-green)';
        else if (v > 120 && v <= 160) bpmValueEl.style.color = 'var(--clr-yellow)';
        else                          bpmValueEl.style.color = 'var(--clr-red)';
    }
}

/* ─── rPPG waveform ──────────────────────────────────────────────── */
function updateWaveformBuffer(samples) {
    waveformHasData = true;
    waveformIdle.classList.add('hidden');
    for (const s of samples) {
        waveformBuffer.push(s);
    }
    waveformBuffer = waveformBuffer.slice(-WAVEFORM_POINTS);
}

function drawWaveform() {
    const rect = rppgCanvas.getBoundingClientRect();
    rppgCanvas.width  = rect.width  || 400;
    rppgCanvas.height = rect.height || 90;
    const W = rppgCanvas.width;
    const H = rppgCanvas.height;

    rppgCtx.clearRect(0, 0, W, H);

    if (!waveformHasData) return;

    const data      = waveformBuffer;
    const validData = data.filter(v => v !== null && isFinite(v));
    if (validData.length < 2) return;

    const min   = Math.min(...validData);
    const max   = Math.max(...validData);
    const range = max - min || 1;
    const pad   = 8;
    const xStep = W / (WAVEFORM_POINTS - 1);

    // Glow pass
    rppgCtx.beginPath();
    rppgCtx.strokeStyle = 'rgba(0, 229, 255, 0.18)';
    rppgCtx.lineWidth   = 5;
    rppgCtx.lineJoin    = 'round';
    rppgCtx.lineCap     = 'round';
    plotWave(data, min, range, W, H, pad, xStep);
    rppgCtx.stroke();

    // Sharp line pass
    rppgCtx.beginPath();
    rppgCtx.strokeStyle = 'rgba(0, 229, 255, 0.85)';
    rppgCtx.lineWidth   = 1.5;
    rppgCtx.lineJoin    = 'round';
    rppgCtx.lineCap     = 'round';
    plotWave(data, min, range, W, H, pad, xStep);
    rppgCtx.stroke();

    // Fill area under curve
    rppgCtx.beginPath();
    plotWave(data, min, range, W, H, pad, xStep);
    rppgCtx.lineTo(W, H);
    rppgCtx.lineTo(0, H);
    rppgCtx.closePath();
    const fillGrad = rppgCtx.createLinearGradient(0, 0, 0, H);
    fillGrad.addColorStop(0, 'rgba(0,229,255,0.14)');
    fillGrad.addColorStop(1, 'rgba(0,229,255,0)');
    rppgCtx.fillStyle = fillGrad;
    rppgCtx.fill();
}

function plotWave(data, min, range, W, H, pad, xStep) {
    let started = false;
    for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (v === null || !isFinite(v)) continue;
        const x = i * xStep;
        const y = H - pad - ((v - min) / range) * (H - pad * 2);
        if (!started) { rppgCtx.moveTo(x, y); started = true; }
        else          { rppgCtx.lineTo(x, y); }
    }
}

/* ─── Signal card update ─────────────────────────────────────────── */
/**
 * score is [0..1] from backend.
 * null/undefined means "not yet available".
 */
function updateSignalCard(cardEl, scoreEl, barEl, score) {
    cardEl.classList.remove('sig-pass', 'sig-warn', 'sig-fail');

    if (score === null || score === undefined) {
        scoreEl.textContent = '—';
        barEl.style.width   = '0%';
        return;
    }

    const safe = clamp(score, 0, 1);
    const pct  = Math.round(safe * 100);
    scoreEl.textContent = `${pct}%`;
    barEl.style.width   = `${pct}%`;

    if (safe >= 0.70)      cardEl.classList.add('sig-pass');
    else if (safe >= 0.45) cardEl.classList.add('sig-warn');
    else                   cardEl.classList.add('sig-fail');
}

/* ─── Trust gauge update ─────────────────────────────────────────── */
/** trust is [0..100] from backend. */
function updateTrustGauge(trust) {
    if (trust === null || trust === undefined) {
        trustScoreEl.textContent          = '—';
        gaugeArc.style.strokeDashoffset   = GAUGE_ARC_LEN;
        trustScoreEl.style.color          = '';
        return;
    }

    const safe   = clamp(trust, 0, 100);
    const offset = GAUGE_ARC_LEN - (safe / 100) * GAUGE_ARC_LEN;
    gaugeArc.style.strokeDashoffset = offset;
    trustScoreEl.textContent = safe.toFixed(1);

    if (safe >= 75)      trustScoreEl.style.color = 'var(--clr-green)';
    else if (safe >= 50) trustScoreEl.style.color = 'var(--clr-yellow)';
    else                 trustScoreEl.style.color = 'var(--clr-red)';
}

/* ─── Verification decision card ─────────────────────────────────── */
/**
 * status_flag: "nominal" | "calibrating" | "low_confidence" | "spoof_suspected"
 *
 * Note: "SIGNALS NOMINAL" is used (not "VERIFIED") because the backend emits
 * this on every frame meeting the threshold — it is per-frame telemetry, not
 * a concluded session decision.
 */
const DECISION_STATES = {
    nominal: {
        cls:    'dec-nominal',
        icon:   '🟢',
        title:  'SIGNALS NOMINAL — LIVE HUMAN CONSISTENT',
        detail: 'All biometric signals are consistent with a live human subject. Biological pulse, acoustic characteristics, and audio-visual synchronization are within expected ranges.',
    },
    calibrating: {
        cls:    'dec-calibrating',
        icon:   '🔵',
        title:  'CALIBRATING',
        detail: 'Collecting biometric samples. Please face the camera directly, keep still, and speak naturally for a few seconds.',
    },
    low_confidence: {
        cls:    'dec-low-confidence',
        icon:   '🟡',
        title:  'LOW CONFIDENCE',
        detail: 'Signal quality is insufficient for a high-confidence decision. Check lighting, camera position, and ensure the microphone is active.',
    },
    spoof_suspected: {
        cls:    'dec-spoof',
        icon:   '🔴',
        title:  'SUSPICIOUS — SIGNALS INCONSISTENT',
        detail: 'One or more biometric signals show anomalies that may indicate synthetic or replayed media. Verification denied.',
    },
};

function updateDecision(statusFlag) {
    const state = DECISION_STATES[statusFlag] || DECISION_STATES.calibrating;
    decisionCard.className     = `decision-card ${state.cls}`;
    decisionIcon.textContent   = state.icon;
    decisionTitle.textContent  = state.title;
    decisionDetail.textContent = state.detail;
}

/* ─── Utility ────────────────────────────────────────────────────── */
function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

/* ─── Boot ───────────────────────────────────────────────────────── */
init();
