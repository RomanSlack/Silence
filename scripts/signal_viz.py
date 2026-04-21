#!/usr/bin/env python3
"""
Live EMG waveform viewer — shows raw signal for ch1 (mentalis) and ch2 (masseter).
Clench your jaw and watch for amplitude bursts above the noise floor.

Usage: sudo ../ml_backend/.venv/bin/python signal_viz.py
Then open http://127.0.0.1:5050/
"""

import json
import sys
import time
import threading
import numpy as np
from flask import Flask, Response, render_template_string

try:
    from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
except ImportError:
    print("Run with venv: sudo ../ml_backend/.venv/bin/python signal_viz.py")
    sys.exit(1)

PORT = "/dev/ttyUSB0"
SAMPLE_RATE = 250
CHUNK_SAMPLES = 25  # 100ms chunks
DISPLAY_CHANNELS = [0, 1]  # ch1=mentalis, ch2=masseter
CH_LABELS = ["CH1 — mentalis (chin)", "CH2 — masseter (jaw hinge)"]
CH_COLORS = ["#ff4444", "#ff8800"]

latest_chunk = {"samples": [[], []], "ts": 0}

def start_board():
    BoardShim.disable_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    while True:
        time.sleep(0.1)
        data = board.get_current_board_data(CHUNK_SAMPLES)
        if data.shape[1] == 0:
            continue
        samples = [data[ch].tolist() for ch in DISPLAY_CHANNELS]
        latest_chunk["samples"] = samples
        latest_chunk["ts"] = time.time()

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
<title>EMG Live</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { background: #0d0d0d; color: #ccc; font-family: monospace; margin: 0; padding: 20px; }
  h2 { margin: 0 0 6px; font-size: 13px; color: #666; font-weight: normal; }
  p  { margin: 0 0 20px; font-size: 11px; color: #444; }
  .card { background: #161616; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .label { font-size: 13px; margin-bottom: 4px; }
  .stat { font-size: 11px; color: #666; margin-bottom: 10px; }
  canvas { width: 100% !important; }
</style>
</head>
<body>
<h2>EMG Live Signal</h2>
<p>Relax jaw → flat baseline. Clench hard → big spike. Good contact = rms&lt;30 at rest, rms&gt;200 on clench.</p>
<div id="cards"></div>
<script>
const HISTORY = 500;
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
  statEl.textContent = 'rms: —';
  statEls.push(statEl);
  const canvas = document.createElement('canvas');
  canvas.height = 120;
  card.append(labelEl, statEl, canvas);
  document.getElementById('cards').appendChild(card);

  charts.push(new Chart(canvas, {
    type: 'line',
    data: {
      labels: Array(HISTORY).fill(''),
      datasets: [{
        data: Array(HISTORY).fill(0),
        borderColor: COLORS[i],
        borderWidth: 1.5,
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
          grid: { color: '#1e1e1e' }
        }
      }
    }
  }));
});

const es = new EventSource('/stream');
es.onmessage = e => {
  const samples = JSON.parse(e.data);
  samples.forEach((chunk, i) => {
    const ds = charts[i].data.datasets[0].data;
    chunk.forEach(v => {
      ds.push(v);
      if (ds.length > HISTORY) ds.shift();
    });
    const arr = ds.filter(v => v !== 0);
    const rms = arr.length ? Math.sqrt(arr.reduce((s,v) => s + v*v, 0) / arr.length) : 0;
    statEls[i].textContent = `rms: ${rms.toFixed(1)} µV`;
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
                yield f"data: {json.dumps(latest_chunk['samples'])}\n\n"
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    print("Connecting to board...")
    t = threading.Thread(target=start_board, daemon=True)
    t.start()
    time.sleep(3)
    print("Open http://127.0.0.1:5050/")
    app.run(host="127.0.0.1", port=5050, threaded=True)
