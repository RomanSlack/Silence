#!/usr/bin/env python3
"""
Live bipolar-mode EMG viewer with 20-115 Hz bandpass.

Wires: electrode A -> N1P, electrode B -> N1N, ~2 cm apart on target muscle.
(No BIAS, no SRB needed — the clone board's BIAS/SRB pins are dead.)

Shows scrolling filtered waveform + rolling RMS gauge.
Rest -> flat, RMS small. Clench -> bursts, RMS spikes.

Usage: sudo ../ml_backend/.venv/bin/python scripts/bipolar_viz.py
Then open http://127.0.0.1:5050/
"""

import json
import sys
import time
import threading
from collections import deque
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from flask import Flask, Response, render_template_string

try:
    from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
except ImportError:
    print("Run with venv: sudo ../ml_backend/.venv/bin/python scripts/bipolar_viz.py")
    sys.exit(1)

PORT = "/dev/ttyUSB0"
FS = 250.0
CHUNK_SAMPLES = 25  # 100 ms chunks

# Bipolar config (no SRB, no BIAS) for CH1 and CH2:
CMDS = ['x1060000X', 'x2060000X']

DISPLAY_CHANNELS = [0, 1]
CH_LABELS = ["CH1 — N1P/N1N", "CH2 — N2P/N2N"]
CH_COLORS = ["#ff4444", "#ff8800"]

sos = butter(4, [20.0, 115.0], btype='band', fs=FS, output='sos')
zi_per_ch = [sosfilt_zi(sos).copy() for _ in DISPLAY_CHANNELS]

# Rolling RMS buffer per channel (last ~0.5 s of filtered samples)
RMS_WIN = int(0.5 * FS)
rms_bufs = [deque(maxlen=RMS_WIN) for _ in DISPLAY_CHANNELS]

latest_chunk = {"samples": [[] for _ in DISPLAY_CHANNELS],
                "rms": [0.0 for _ in DISPLAY_CHANNELS],
                "ts": 0}

def start_board():
    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()

    for cmd in CMDS:
        board.config_board(cmd)
        time.sleep(0.3)

    board.start_stream()
    emg_rows = BoardShim.get_emg_channels(BoardIds.CYTON_BOARD.value)

    while True:
        time.sleep(0.1)
        data = board.get_current_board_data(CHUNK_SAMPLES)
        if data.shape[1] == 0:
            continue
        filtered_per_ch = []
        for i, ch in enumerate(DISPLAY_CHANNELS):
            raw = data[emg_rows[ch]].astype(np.float64)
            raw -= raw.mean()  # strip DC per chunk
            filt, zi_per_ch[i] = sosfilt(sos, raw, zi=zi_per_ch[i])
            filtered_per_ch.append(filt.tolist())
            rms_bufs[i].extend(filt.tolist())
            arr = np.asarray(rms_bufs[i])
            latest_chunk["rms"][i] = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
        latest_chunk["samples"] = filtered_per_ch
        latest_chunk["ts"] = time.time()

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
<title>EMG Bipolar Live</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { background: #0d0d0d; color: #ccc; font-family: monospace; margin: 0; padding: 20px; }
  h2 { margin: 0 0 6px; font-size: 13px; color: #666; font-weight: normal; }
  p  { margin: 0 0 20px; font-size: 11px; color: #444; }
  .card { background: #161616; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .label { font-size: 13px; margin-bottom: 4px; }
  .stat { font-size: 11px; color: #888; margin-bottom: 10px; }
  .state-rest   { color: #555; }
  .state-maybe  { color: #ffcc00; }
  .state-clench { color: #44ff88; font-weight: bold; }
  canvas { width: 100% !important; }
</style>
</head>
<body>
<h2>EMG Bipolar — 20-115 Hz bandpass</h2>
<p>Rest -> flat baseline, RMS &lt; 30 µV. Clench -> bursts, RMS &gt; 100 µV.</p>
<div id="cards"></div>
<script>
const HISTORY = 750;  // ~3 s @ 250 Hz
const COLORS = {{ colors|safe }};
const LABELS = {{ labels|safe }};
const charts = [], statEls = [];

LABELS.forEach((lbl, i) => {
  const card = document.createElement('div');
  card.className = 'card';
  const labelEl = document.createElement('div');
  labelEl.className = 'label';
  labelEl.style.color = COLORS[i];
  labelEl.textContent = lbl;
  const statEl = document.createElement('div');
  statEl.className = 'stat';
  statEl.innerHTML = 'rms: — &nbsp;&nbsp;·&nbsp;&nbsp; <span class="state-rest">rest</span>';
  statEls.push(statEl);
  const canvas = document.createElement('canvas');
  canvas.height = 140;
  card.append(labelEl, statEl, canvas);
  document.getElementById('cards').appendChild(card);

  charts.push(new Chart(canvas, {
    type: 'line',
    data: {
      labels: Array(HISTORY).fill(''),
      datasets: [{
        data: Array(HISTORY).fill(0),
        borderColor: COLORS[i],
        borderWidth: 1.2,
        pointRadius: 0,
        fill: false,
        tension: 0
      }]
    },
    options: {
      animation: false,
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#555', font: { size: 10 } },
          grid: { color: '#1e1e1e' },
          suggestedMin: -300,
          suggestedMax: 300
        }
      }
    }
  }));
});

const THRESHOLDS = [
  { active: 175, maybe: 140 },   // CH1 masseter: rest ~126, clench ~220
  { active: 60,  maybe: 30  },   // CH2 orbicularis: rest ~15-20, purse ~80+
];
function stateLabel(rms, i) {
  const t = THRESHOLDS[i] || THRESHOLDS[0];
  if (rms > t.active) return '<span class="state-clench">ACTIVE</span>';
  if (rms > t.maybe)  return '<span class="state-maybe">?</span>';
  return '<span class="state-rest">rest</span>';
}

const es = new EventSource('/stream');
es.onmessage = e => {
  const msg = JSON.parse(e.data);
  msg.samples.forEach((chunk, i) => {
    const ds = charts[i].data.datasets[0].data;
    chunk.forEach(v => {
      ds.push(v);
      if (ds.length > HISTORY) ds.shift();
    });
    const rms = msg.rms[i];
    statEls[i].innerHTML = `rms: ${rms.toFixed(1)} µV &nbsp;&nbsp;·&nbsp;&nbsp; ${stateLabel(rms, i)}`;
    charts[i].update('none');
  });
};
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML,
        colors=json.dumps(CH_COLORS),
        labels=json.dumps(CH_LABELS))

@app.route("/stream")
def stream():
    def gen():
        last_ts = 0
        while True:
            time.sleep(0.05)
            if latest_chunk["ts"] != last_ts:
                last_ts = latest_chunk["ts"]
                yield f"data: {json.dumps({'samples': latest_chunk['samples'], 'rms': latest_chunk['rms']})}\n\n"
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    print("Connecting to board and configuring bipolar mode for CH1+CH2...")
    t = threading.Thread(target=start_board, daemon=True)
    t.start()
    time.sleep(4)
    print("Open http://127.0.0.1:5050/")
    app.run(host="127.0.0.1", port=5050, threaded=True)
