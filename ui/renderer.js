const WS_URL = 'ws://localhost:8765';
let ws;
let reconnectTimer;
let autoScroll = true;

// DOM refs
const indicator = document.getElementById('indicator');
const elapsed = document.getElementById('elapsed');
const vuFill = document.getElementById('vu-fill');
const chunks = document.getElementById('chunks');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');
const analysis = document.getElementById('analysis');
const sourceSelect = document.getElementById('source');
const btnPause = document.getElementById('btn-pause');
const btnSave = document.getElementById('btn-save');

// Track scroll position
transcript.addEventListener('scroll', () => {
  const { scrollTop, scrollHeight, clientHeight } = transcript;
  autoScroll = scrollHeight - scrollTop - clientHeight < 50;
});

function connect() {
  status.textContent = 'Connecting...';
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    status.textContent = 'Connected';
    if (reconnectTimer) {
      clearInterval(reconnectTimer);
      reconnectTimer = null;
    }
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'state') {
      updateUI(data);
    } else if (data.type === 'saved') {
      showToast(data.path ? `Saved: ${data.path}` : 'Nothing to save');
    }
  };

  ws.onclose = () => {
    status.textContent = 'Disconnected. Reconnecting...';
    if (!reconnectTimer) {
      reconnectTimer = setInterval(connect, 2000);
    }
  };

  ws.onerror = () => {
    ws.close();
  };
}

function updateUI(data) {
  // Indicator
  if (data.paused) {
    indicator.className = 'paused';
    indicator.textContent = 'PAUSED';
  } else if (data.is_speaking) {
    indicator.className = 'speaking';
    indicator.textContent = 'REC';
  } else {
    indicator.className = 'idle';
    indicator.textContent = 'IDLE';
  }

  // Elapsed
  elapsed.textContent = data.elapsed_str;

  // VU meter
  const level = Math.min(data.audio_level * 500, 100);
  vuFill.style.width = level + '%';
  if (level > 60) {
    vuFill.style.background = '#ef4444';
  } else if (level > 30) {
    vuFill.style.background = '#eab308';
  } else {
    vuFill.style.background = '#22c55e';
  }

  // Chunks
  chunks.textContent = data.chunks + ' chunks';

  // Status
  status.textContent = data.status;

  // Source selector
  if (sourceSelect.value !== data.source) {
    sourceSelect.value = data.source;
  }

  // Pause button
  btnPause.textContent = data.paused ? 'Resume' : 'Pause';
  btnPause.className = data.paused ? 'active' : '';

  // Transcript
  updateTranscript(data.transcript);

  // Analysis
  updateAnalysis(data.analysis);
}

function updateTranscript(entries) {
  // Only rebuild if entry count changed
  if (transcript.childElementCount === entries.length) return;

  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    const div = document.createElement('div');
    div.className = 'transcript-entry';
    div.innerHTML = `<span class="transcript-time">${entry.time}</span><span class="transcript-text">${escapeHtml(entry.text)}</span>`;
    fragment.appendChild(div);
  }

  transcript.innerHTML = '';
  transcript.appendChild(fragment);

  if (autoScroll) {
    transcript.scrollTop = transcript.scrollHeight;
  }
}

function updateAnalysis(data) {
  if (!data) return;

  let html = '';

  // Sentiment badge
  const sentimentClass = 'sentiment-' + data.sentiment;
  html += `<div id="sentiment-badge" class="${sentimentClass}">${data.sentiment.toUpperCase()}</div>`;

  // Sections
  const sections = [
    { title: 'Key Points', key: 'key_points', cls: 'key-points' },
    { title: 'Objections', key: 'objections', cls: 'objections' },
    { title: 'Suggestions', key: 'suggested_responses', cls: 'suggestions' },
    { title: 'Questions', key: 'questions', cls: 'questions' },
    { title: 'Action Items', key: 'action_items', cls: 'action-items' },
  ];

  for (const section of sections) {
    const items = data[section.key];
    if (items && items.length > 0) {
      html += `<div class="analysis-section">`;
      html += `<div class="analysis-section-title ${section.cls}">${section.title}</div>`;
      for (const item of items) {
        html += `<div class="analysis-item">${escapeHtml(item)}</div>`;
      }
      html += `</div>`;
    }
  }

  // Summary
  if (data.summary) {
    html += `<div class="analysis-summary">${escapeHtml(data.summary)}</div>`;
  }

  analysis.innerHTML = html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'save-toast';
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// Controls
sourceSelect.addEventListener('change', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'source', value: sourceSelect.value }));
  }
});

btnPause.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'pause' }));
  }
});

btnSave.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: 'save' }));
  }
});

// Start
connect();
