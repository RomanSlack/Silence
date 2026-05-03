// Silence recorder — client state + keybindings.
// Spacebar: record. U: undo last trial. Arrows: change target word. R: toggle round-robin.

(() => {
  const durationMs = Math.round(window.__CFG__.duration * 1000);
  const countdownMs = 1000;
  const vocab = window.__CFG__.vocab;

  const wordEl = document.getElementById("word");
  const stateEl = document.getElementById("state-label");
  const countdownEl = document.getElementById("countdown");
  const hintEl = document.getElementById("hint");
  const signalEl = document.getElementById("signal-readout");
  const gridEl = document.getElementById("grid");
  const totalEl = document.getElementById("total");
  const toastEl = document.getElementById("toast");
  const modeEl = document.getElementById("mode");

  let counts = {};
  let currentIdx = 0;
  let isRecording = false;
  let roundRobin = true;

  const modeLabel = () => roundRobin ? "round-robin" : "focus";

  function toast(msg, err = false) {
    toastEl.textContent = msg;
    toastEl.classList.toggle("err", err);
    toastEl.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toastEl.classList.remove("show"), 1600);
  }

  function renderGrid() {
    const target = Math.max(...Object.values(counts), 5);  // guess target based on best word
    const desired = Math.max(target, 10);
    gridEl.innerHTML = "";
    vocab.forEach((w, i) => {
      const cell = document.createElement("div");
      cell.className = "cell";
      if (i === currentIdx) cell.classList.add("active");
      if ((counts[w] || 0) >= desired) cell.classList.add("done");
      cell.innerHTML = `<span class="w">${w}</span><span class="n">${counts[w] || 0}</span>`;
      cell.onclick = () => { currentIdx = i; updateWord(); };
      gridEl.appendChild(cell);
    });
    totalEl.textContent = Object.values(counts).reduce((a, b) => a + b, 0);
  }

  function updateWord() {
    wordEl.textContent = vocab[currentIdx];
    wordEl.className = "word";
    renderGrid();
  }

  async function refreshState() {
    const r = await fetch("/api/state");
    const s = await r.json();
    counts = s.counts;
    renderGrid();
  }

  async function recordOne() {
    if (isRecording) return;
    isRecording = true;
    const word = vocab[currentIdx];
    wordEl.className = "word armed";
    stateEl.textContent = "get ready";
    // countdown
    for (let s = Math.round(countdownMs / 1000); s > 0; s--) {
      countdownEl.textContent = s;
      await sleep(1000);
    }
    countdownEl.textContent = "";
    wordEl.className = "word recording";
    stateEl.textContent = "recording";
    const t0 = performance.now();
    try {
      const res = await fetch("/api/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word }),
      });
      const data = await res.json();
      const t = Math.round(performance.now() - t0);
      if (!data.ok) {
        toast(data.error || "record failed", true);
      } else {
        counts = data.counts;
        const rms = data.rms.toFixed(3);
        const peak = data.peak.toFixed(1);
        signalEl.textContent = `rms=${rms}  peak=${peak}  dt=${t}ms`;
        wordEl.className = "word saved";
        stateEl.textContent = `saved • trial ${data.trial_idx + 1}`;
        if (roundRobin) currentIdx = (currentIdx + 1) % vocab.length;
        await sleep(350);
      }
    } catch (e) {
      toast("network error: " + e.message, true);
    } finally {
      isRecording = false;
      stateEl.textContent = "";
      updateWord();
    }
  }

  async function undo() {
    if (isRecording) return;
    const res = await fetch("/api/undo", { method: "POST" });
    const data = await res.json();
    if (!data.ok) { toast(data.error || "undo failed", true); return; }
    counts = data.counts;
    toast(`undid ${data.removed.word} #${data.removed.trial_idx + 1}`);
    renderGrid();
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") {
      e.preventDefault();
      recordOne();
    } else if (e.key === "u" || e.key === "U") {
      undo();
    } else if (e.key === "ArrowRight") {
      currentIdx = (currentIdx + 1) % vocab.length;
      updateWord();
    } else if (e.key === "ArrowLeft") {
      currentIdx = (currentIdx - 1 + vocab.length) % vocab.length;
      updateWord();
    } else if (e.key === "r" || e.key === "R") {
      roundRobin = !roundRobin;
      modeEl.textContent = modeLabel();
      toast(`mode: ${modeLabel()}`);
    }
  });

  document.getElementById("btn-record").onclick = recordOne;
  document.getElementById("btn-undo").onclick = undo;

  // session picker
  const sessionNameEl = document.getElementById("session-name");
  const dropdownEl    = document.getElementById("session-dropdown");

  document.getElementById("btn-new-session").onclick = async () => {
    const r = await fetch("/api/new-session", { method: "POST" });
    const d = await r.json();
    if (d.ok) { sessionNameEl.textContent = d.session; counts = {}; renderGrid(); toast("new session: " + d.session); }
  };

  document.getElementById("btn-sessions").onclick = async () => {
    if (!dropdownEl.hidden) { dropdownEl.hidden = true; return; }
    const r = await fetch("/api/sessions");
    const d = await r.json();
    dropdownEl.innerHTML = "";
    d.sessions.forEach(name => {
      const el = document.createElement("div");
      el.className = "session-item" + (name === d.current ? " current" : "");
      el.textContent = name;
      el.onclick = async () => {
        dropdownEl.hidden = true;
        const res = await fetch("/api/load-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session: name }),
        });
        const data = await res.json();
        if (data.ok) { sessionNameEl.textContent = data.session; await refreshState(); updateWord(); toast("loaded: " + data.session); }
      };
      dropdownEl.appendChild(el);
    });
    dropdownEl.hidden = false;
  };

  document.addEventListener("click", e => {
    if (!dropdownEl.hidden && !dropdownEl.contains(e.target) && e.target.id !== "btn-sessions")
      dropdownEl.hidden = true;
  });

  modeEl.textContent = modeLabel();
  refreshState().then(updateWord);
})();
