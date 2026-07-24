/**
 * dashboard.js — DD-Net Dashboard Logic
 * ======================================
 * Connects to /stats_stream (SSE) and updates all UI elements:
 *   - EAR semicircular gauge (canvas)
 *   - EAR history line chart (canvas)
 *   - Status circle (CSS-based)
 *   - Telemetry cards
 *   - Alert log
 *   - Calibration flow
 */

// ── DOM refs ─────────────────────────────────────────────────────────────────
const els = {
  // Header
  globalStatus:   document.getElementById('global-status'),
  statusIcon:     document.getElementById('status-icon'),
  statusText:     document.getElementById('status-text'),
  sessionTimer:   document.getElementById('session-timer'),

  // Badges
  badgeEar:       document.getElementById('badge-ear'),
  badgeEarVal:    document.getElementById('badge-ear-val'),
  badgeFps:       document.getElementById('badge-fps'),
  threshDisplay:  document.getElementById('threshold-display'),
  threshLabel:    document.getElementById('thresh-label'),
  telemThreshold: document.getElementById('telem-threshold'),

  // EAR readout (telemetry card)
  earValue:       document.getElementById('ear-value'),

  // Chart
  chartCanvas:    document.getElementById('ear-chart'),

  // Metrics
  metricBlinks:   document.getElementById('metric-blinks'),
  metricFps:      document.getElementById('metric-fps'),
  metricSession:  document.getElementById('metric-session'),

  // Status circle
  statusRing:     document.getElementById('status-ring'),
  statusMain:     document.getElementById('status-main'),
  statusSub:      document.getElementById('status-sub'),
  statusNote:     document.getElementById('status-note'),

  // Video panel
  videoPanel:     document.getElementById('video-panel'),
  videoOverlay:   document.getElementById('video-alert-overlay'),
  videoStatus:    document.getElementById('video-status-text'),
  videoLoading:   document.getElementById('video-loading'),

  // Alert banner
  alertBanner:    document.getElementById('alert-banner'),

  // Log
  logList:        document.getElementById('log-list'),

  // Calibration
  calibBtn:       document.getElementById('calib-btn'),
  calibHint:      document.getElementById('calib-hint'),
  calibCountdown: document.getElementById('calib-countdown'),
};

// ── Globals ───────────────────────────────────────────────────────────────────
let liveThreshold = 0.20;
let earHistory    = [];
const EAR_MAX     = 0.42;
const EAR_LOW     = 0.16;

// ── EAR History Chart ─────────────────────────────────────────────────────────
function drawChart(history, threshold) {
  const canvas = els.chartCanvas;
  if (!canvas) return;
  const thresh = threshold ?? liveThreshold;
  const dpr    = window.devicePixelRatio || 1;
  canvas.width  = canvas.offsetWidth  * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W    = canvas.offsetWidth;
  const H    = canvas.offsetHeight;
  const pad  = { top: 8, bottom: 20, left: 32, right: 8 };
  const cW   = W - pad.left - pad.right;
  const cH   = H - pad.top  - pad.bottom;
  const data = history.length > 1 ? history : [0.30];

  const toY = v  => pad.top  + cH - (v / EAR_MAX) * cH;
  const toX = i  => pad.left + (i / (data.length - 1)) * cW;

  ctx.clearRect(0, 0, W, H);

  // Y-axis grid lines
  [0.10, 0.20, 0.30, 0.40].forEach(v => {
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cW, y);
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.fillStyle = '#484f58';
    ctx.font = `${9}px JetBrains Mono, monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), pad.left - 4, y + 3);
  });
  ctx.textAlign = 'left';

  // Threshold zone fill
  const ty = toY(thresh);
  ctx.fillStyle = 'rgba(248,81,73,0.06)';
  ctx.fillRect(pad.left, ty, cW, toY(0) - ty);

  // Threshold dashed line
  ctx.beginPath();
  ctx.moveTo(pad.left, ty);
  ctx.lineTo(pad.left + cW, ty);
  ctx.strokeStyle = 'rgba(210,153,34,0.5)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(210,153,34,0.7)';
  ctx.font = `9px JetBrains Mono`;
  ctx.fillText(thresh.toFixed(2), pad.left + 4, ty - 3);

  // Area fill
  if (data.length > 1) {
    const grad = ctx.createLinearGradient(0, pad.top, 0, H);
    grad.addColorStop(0,   'rgba(63,185,80,0.30)');
    grad.addColorStop(0.6, 'rgba(63,185,80,0.06)');
    grad.addColorStop(1,   'rgba(63,185,80,0.00)');

    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i = 1; i < data.length; i++) {
      const mx = (toX(i-1) + toX(i)) / 2;
      const my = (toY(data[i-1]) + toY(data[i])) / 2;
      ctx.quadraticCurveTo(toX(i-1), toY(data[i-1]), mx, my);
    }
    ctx.lineTo(toX(data.length-1), toY(data[data.length-1]));
    ctx.lineTo(toX(data.length-1), H - pad.bottom);
    ctx.lineTo(toX(0), H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // Line
  if (data.length > 1) {
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(data[0]));
    for (let i = 1; i < data.length; i++) {
      const mx = (toX(i-1) + toX(i)) / 2;
      const my = (toY(data[i-1]) + toY(data[i])) / 2;
      ctx.quadraticCurveTo(toX(i-1), toY(data[i-1]), mx, my);
    }
    ctx.lineTo(toX(data.length-1), toY(data[data.length-1]));
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth   = 1.8;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';
    ctx.shadowColor = '#3fb950';
    ctx.shadowBlur  = 5;
    ctx.stroke();
    ctx.shadowBlur  = 0;

    // Latest dot
    const lx = toX(data.length-1);
    const ly = toY(data[data.length-1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fillStyle   = '#3fb950';
    ctx.shadowColor = '#3fb950';
    ctx.shadowBlur  = 8;
    ctx.fill();
    ctx.shadowBlur  = 0;
  }

  // X-axis labels
  ctx.fillStyle = '#484f58';
  ctx.font      = '9px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Past', pad.left, H - 4);
  ctx.textAlign = 'right';
  ctx.fillText('Now', W - pad.right, H - 4);
}

// ── Alert Log ─────────────────────────────────────────────────────────────────
let _lastLogKey = '';

function updateLog(logs) {
  if (!logs || logs.length === 0) {
    els.logList.innerHTML = '<div class="log-empty">No events recorded yet.</div>';
    return;
  }
  const key = JSON.stringify(logs);
  if (key === _lastLogKey) return;
  _lastLogKey = key;

  els.logList.innerHTML = logs.map(e => {
    const cls   = e.event === 'DROWSY'   ? 'drowsy'   :
                  e.event === 'CAUTION'  ? 'caution'  : 'recovered';
    const label = e.event === 'DROWSY'   ? 'DROWSINESS' :
                  e.event === 'CAUTION'  ? 'CAUTION'    : 'RECOVERED';
    return `<div class="log-entry ${cls}">
      <div class="log-dot"></div>
      <span class="log-time">${e.time}</span>
      <span class="log-event">${label}</span>
      <span class="log-ear">EAR ${e.ear.toFixed(3)}</span>
    </div>`;
  }).join('');
}

// ── Status Updates ────────────────────────────────────────────────────────────
function updateStatus(alert, warning, faceFound) {
  const { globalStatus, statusIcon, statusText, statusRing,
          statusMain, statusSub, statusNote, videoStatus,
          videoPanel, videoOverlay, alertBanner } = els;

  if (!faceFound) {
    globalStatus.className = 'gs-noface';
    statusIcon.className   = 'gs-dot';
    statusText.textContent = 'NO FACE';

    statusRing.className   = 'status-ring ring-noface';
    statusMain.textContent = 'NO FACE';
    statusSub.textContent  = 'DETECTED';
    statusNote.textContent = 'Move closer to camera';

    videoStatus.className   = 'vf-noface';
    videoStatus.textContent = 'No Face Detected';

    videoPanel.classList.remove('alert-mode', 'warn-mode');
    videoOverlay.classList.remove('visible');
    alertBanner.classList.remove('visible');
    return;
  }

  if (alert) {
    // ── DROWSY ──────────────────────────────────────────────
    globalStatus.className = 'gs-drowsy';
    statusIcon.className   = 'gs-dot blink';
    statusText.textContent = 'DROWSINESS DETECTED';

    statusRing.className   = 'status-ring ring-drowsy';
    statusMain.textContent = 'ALERT:';
    statusSub.textContent  = 'DROWSY';
    statusNote.textContent = 'Please pull over safely';

    videoStatus.className   = 'vf-drowsy';
    videoStatus.textContent = 'Alert: Drowsiness Detected';

    videoPanel.classList.add('alert-mode');
    videoPanel.classList.remove('warn-mode');
    videoOverlay.classList.add('visible');
    alertBanner.classList.add('visible');

  } else if (warning) {
    // ── WARNING ────────────────────────────────────────────
    globalStatus.className = 'gs-warning';
    statusIcon.className   = 'gs-dot';
    statusText.textContent = 'CAUTION';

    statusRing.className   = 'status-ring ring-warning';
    statusMain.textContent = 'CAUTION:';
    statusSub.textContent  = 'WARNING';
    statusNote.textContent = 'Eyes closing — stay alert';

    videoStatus.className   = 'vf-warning';
    videoStatus.textContent = 'Caution: Eyes Closing';

    videoPanel.classList.add('warn-mode');
    videoPanel.classList.remove('alert-mode');
    videoOverlay.classList.remove('visible');
    alertBanner.classList.remove('visible');

  } else {
    // ── AWAKE ──────────────────────────────────────────────
    globalStatus.className = 'gs-normal';
    statusIcon.className   = 'gs-dot';
    statusText.textContent = 'DRIVER ALERT';

    statusRing.className   = 'status-ring';
    statusMain.textContent = 'NORMAL:';
    statusSub.textContent  = 'AWAKE';
    statusNote.textContent = 'No drowsiness detected';

    videoStatus.className   = 'vf-normal';
    videoStatus.textContent = 'Normal: Awake';

    videoPanel.classList.remove('alert-mode', 'warn-mode');
    videoOverlay.classList.remove('visible');
    alertBanner.classList.remove('visible');
  }
}

// ── Calibration ───────────────────────────────────────────────────────────────
function startCalibration() {
  const btn = els.calibBtn;
  if (btn.disabled) return;

  fetch('/calibrate', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.status === 'already_calibrating') return;
      btn.disabled = true;
      btn.classList.add('calibrating');
      document.getElementById('calib-text').textContent = 'Calibrating...';
      els.calibHint.classList.remove('hidden');
      els.calibCountdown.classList.remove('hidden');
    })
    .catch(err => console.error('Calibration error:', err));
}

function updateCalibrationUI(isCalibrating, countdown, threshold) {
  const btn = els.calibBtn;
  if (!btn) return;

  if (isCalibrating) {
    btn.disabled = true;
    btn.classList.add('calibrating');
    btn.classList.remove('success');
    document.getElementById('calib-text').textContent = 'Calibrating...';
    els.calibHint.classList.remove('hidden');
    els.calibCountdown.classList.remove('hidden');
    els.calibCountdown.textContent = countdown;
  } else {
    if (btn.classList.contains('calibrating')) {
      btn.classList.remove('calibrating');
      btn.classList.add('success');
      document.getElementById('calib-icon').textContent = '\u2713';
      document.getElementById('calib-text').textContent =
        'Done! Threshold: ' + threshold.toFixed(3);
      els.calibHint.classList.add('hidden');
      els.calibCountdown.classList.add('hidden');
      setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove('success');
        document.getElementById('calib-icon').innerHTML = '&#9678;';
        document.getElementById('calib-text').textContent = 'Calibrate for My Eyes';
      }, 3000);
    } else if (!btn.classList.contains('success')) {
      btn.disabled = false;
      els.calibHint.classList.add('hidden');
      els.calibCountdown.classList.add('hidden');
    }
  }
}

// ── Loading Overlay ───────────────────────────────────────────────────────────
let _loadingHidden = false;

function hideLoadingOverlay() {
  if (_loadingHidden) return;
  _loadingHidden = true;
  const el = els.videoLoading;
  if (el) {
    el.style.transition = 'opacity 0.5s ease';
    el.style.opacity    = '0';
    setTimeout(() => el.style.display = 'none', 500);
  }
}

// ── SSE Connection ────────────────────────────────────────────────────────────
function connectSSE() {
  const sse = new EventSource('/stats_stream');

  sse.onmessage = function(e) {
    hideLoadingOverlay();

    const d = JSON.parse(e.data);

    // EAR
    const ear = d.ear || 0;
    if (els.earValue)   els.earValue.textContent   = ear.toFixed(3);
    if (els.badgeEar)   els.badgeEar.textContent   = `EAR ${ear.toFixed(3)}`;
    if (els.badgeEarVal) els.badgeEarVal.textContent = `EAR ${ear.toFixed(3)}`;

    // Live threshold
    if (d.ear_threshold !== undefined) {
      liveThreshold = d.ear_threshold;
      const t = d.ear_threshold.toFixed(3);
      if (els.threshDisplay)  els.threshDisplay.textContent  = t;
      if (els.threshLabel)    els.threshLabel.textContent    = `| ${t}`;
      if (els.telemThreshold) els.telemThreshold.textContent = t;
    }

    // Calibration UI
    updateCalibrationUI(
      d.is_calibrating  || false,
      d.calib_countdown || 0,
      d.ear_threshold   || liveThreshold
    );

    // Telemetry
    if (els.metricBlinks)  els.metricBlinks.textContent  = d.blinks  || 0;
    if (els.metricFps)     els.metricFps.textContent     = (d.fps    || 0).toFixed(1);
    if (els.metricSession) els.metricSession.textContent = d.session || '00:00:00';
    if (els.sessionTimer)  els.sessionTimer.textContent  = d.session || '00:00:00';
    if (els.badgeFps)      els.badgeFps.textContent      = `${(d.fps || 0).toFixed(0)} FPS`;

    // Status (pass warning state)
    updateStatus(d.alert, d.warning || false, d.face_found);

    // Chart only (no gauge)
    if (d.ear_history) {
      earHistory = d.ear_history;
      drawChart(earHistory, liveThreshold);
    }

    // Log
    updateLog(d.alert_log);
  };

  sse.onerror = function() {
    console.warn('SSE disconnected -- retrying in 2s');
    sse.close();
    setTimeout(connectSSE, 2000);
  };
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Initial placeholder draw
  drawChart([0.30], liveThreshold);

  // Redraw chart on window resize
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      drawChart(earHistory.length ? earHistory : [0.30], liveThreshold);
    }, 100);
  });

  connectSSE();
  setTimeout(hideLoadingOverlay, 3000);
});
