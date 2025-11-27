from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from stable_baselines3 import PPO

# Constants for the board layout
W, H = 10, 10


def _resolve_static_root(static_root: Optional[str | Path]) -> Path:
    """Figure out where to serve static assets from (defaults to repo root)."""
    if static_root is None:
        return Path(__file__).resolve().parent.parent.parent
    return Path(static_root).resolve()


def create_app(model_path: str, static_root: Optional[str | Path] = None) -> Flask:
    """Create a Flask app that serves the routing AI."""
    static_root_path = _resolve_static_root(static_root)
    static_root_str = str(static_root_path)

    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        raise RuntimeError(f"Could not find model file '{model_file}'.")

    print(f"Loading AI Model from {model_file}...")
    try:
        model = PPO.load(str(model_file))
        print("Model loaded successfully!")
    except Exception as exc:
        raise RuntimeError(f"Could not load model from '{model_file}': {exc}") from exc

    app = Flask(__name__, static_folder=static_root_str)

    @app.route("/")
    def index():
        return send_from_directory(static_root_str, "index.html")

    @app.route("/<path:path>")
    def serve_static(path):
        return send_from_directory(static_root_str, path)

    @app.route("/get_action", methods=["POST"])
    def get_action():
        data = request.json or {}

        try:
            board = np.array(data["board"], dtype=np.uint8).reshape(H, W)
            directions = np.array(data["directions"], dtype=np.uint8).reshape(H, W)
        except Exception as exc:
            raise RuntimeError(f"Invalid payload for /get_action: {exc}") from exc

        # edit_mask mirrors the board because the AI can edit tiles with pieces on them
        edit_mask = board.copy()
        obs = {"board": board, "directions": directions, "edit_mask": edit_mask}

        action, _ = model.predict(obs, deterministic=True)
        return jsonify({"new_directions": action.tolist()})

    return app


def start_route_ai_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    debug: bool = True,
    static_root: Optional[str | Path] = None,
) -> None:
    """Start the routing AI Flask server."""
    app = create_app(model_path=model_path, static_root=static_root)
    print(f"Starting AI Game Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
