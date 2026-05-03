"""Flask recording UI. Run:

    cd recorder
    ../ml_backend/.venv/bin/python app.py

Opens on http://127.0.0.1:5001 . Reads vocab from ../ml_backend/vocab/mvp_20words.txt
by default, writes sessions to ./data/sessions/{timestamp}/.

Hardware is currently mocked (MockBoard). When Cyton firmware lands, switch
BOARD_KIND to "cyton" and implement CytonBoard in board.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

from board import make_board
from session import Session, new_session_name


HERE = Path(__file__).resolve().parent
DEFAULT_VOCAB = HERE.parent / "ml_backend" / "vocab" / "mvp_20words.txt"
DATA_DIR = HERE / "data" / "sessions"


def load_vocab(path: Path) -> list[str]:
    return [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]


def create_app(session: Session, board, trial_duration_sec: float) -> Flask:
    app = Flask(__name__, template_folder=str(HERE / "templates"),
                static_folder=str(HERE / "static"))
    app.config["SESSION"] = session
    app.config["BOARD"] = board
    app.config["TRIAL_DURATION_SEC"] = trial_duration_sec

    def sess() -> Session:
        return app.config["SESSION"]

    @app.route("/")
    def index():
        return redirect(url_for("record"))

    @app.route("/record")
    def record():
        s = sess()
        return render_template(
            "record.html",
            session_name=s.name,
            vocab=s.vocab,
            duration=trial_duration_sec,
            sample_rate=s.sample_rate,
            n_channels=s.n_channels,
        )

    @app.route("/api/state")
    def api_state():
        s = sess()
        return jsonify({
            "session": s.name,
            "vocab": s.vocab,
            "counts": s.counts(),
            "total": s.total(),
            "duration": trial_duration_sec,
            "sample_rate": s.sample_rate,
            "n_channels": s.n_channels,
            "board": type(board).__name__,
        })

    @app.route("/api/record", methods=["POST"])
    def api_record():
        s = sess()
        word = (request.json or {}).get("word", "").strip().lower()
        if word not in s.vocab:
            return jsonify({"error": f"bad word: {word!r}"}), 400
        signal = board.capture(trial_duration_sec)
        trial = s.save_trial(word, signal, trial_duration_sec)
        rms = float((signal ** 2).mean() ** 0.5)
        peak = float(abs(signal).max())
        return jsonify({
            "ok": True,
            "word": trial.word,
            "trial_idx": trial.trial_idx,
            "path": trial.path,
            "counts": s.counts(),
            "total": s.total(),
            "rms": rms,
            "peak": peak,
        })

    @app.route("/api/undo", methods=["POST"])
    def api_undo():
        s = sess()
        trial = s.undo_last()
        if trial is None:
            return jsonify({"ok": False, "error": "nothing to undo"}), 400
        return jsonify({
            "ok": True,
            "removed": {"word": trial.word, "trial_idx": trial.trial_idx},
            "counts": s.counts(),
            "total": s.total(),
        })

    @app.route("/api/sessions")
    def api_sessions():
        dirs = sorted(
            [p.name for p in DATA_DIR.iterdir() if p.is_dir() and (p / "manifest.json").exists()],
            reverse=True,
        )
        return jsonify({"sessions": dirs, "current": sess().name})

    @app.route("/api/new-session", methods=["POST"])
    def api_new_session():
        s = sess()
        new_sess = Session(
            root=DATA_DIR, name=new_session_name(),
            vocab=s.vocab, sample_rate=s.sample_rate, n_channels=s.n_channels,
        )
        app.config["SESSION"] = new_sess
        return jsonify({"ok": True, "session": new_sess.name})

    @app.route("/api/load-session", methods=["POST"])
    def api_load_session():
        s = sess()
        name = (request.json or {}).get("session", "").strip()
        if not name or not (DATA_DIR / name / "manifest.json").exists():
            return jsonify({"error": "session not found"}), 404
        loaded = Session(
            root=DATA_DIR, name=name,
            vocab=s.vocab, sample_rate=s.sample_rate, n_channels=s.n_channels,
        )
        app.config["SESSION"] = loaded
        return jsonify({"ok": True, "session": loaded.name})

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--session", default=None, help="session name; default = timestamp")
    ap.add_argument("--sample-rate", type=float, default=1000.0)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--duration", type=float, default=2.0, help="seconds per trial")
    ap.add_argument("--board", default="mock", choices=["mock", "cyton"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    session_name = args.session or new_session_name()
    session = Session(
        root=DATA_DIR, name=session_name, vocab=vocab,
        sample_rate=args.sample_rate, n_channels=args.channels,
    )
    board = make_board(args.board, sample_rate=args.sample_rate, n_channels=args.channels)
    board.start()

    print(f"session: {session.dir}")
    print(f"vocab  : {len(vocab)} words")
    print(f"board  : {type(board).__name__} @ {args.sample_rate} Hz × {args.channels} ch")
    print(f"open   : http://{args.host}:{args.port}/")

    app = create_app(session, board, args.duration)
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        board.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
