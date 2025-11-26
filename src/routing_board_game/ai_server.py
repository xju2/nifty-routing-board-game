import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from stable_baselines3 import PPO

# Initialize Flask
app = Flask(__name__, static_folder=".")

# Constants
W, H = 10, 10
MODEL_PATH = (
    "/media/DataOcean/code/nifty-routing-board-game/ppo_router_agent_single_file.zip"
)

print(f"Loading AI Model from {MODEL_PATH}...")
try:
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)


@app.route("/get_action", methods=["POST"])
def get_action():
    data = request.json

    # 1. Parse Input from Browser
    # Browser sends flat lists, we reshape to (10, 10)
    board = np.array(data["board"], dtype=np.uint8).reshape(H, W)
    directions = np.array(data["directions"], dtype=np.uint8).reshape(H, W)

    # 2. Construct Observation for AI
    # We set edit_mask = board because the AI is allowed to edit arrows under pieces
    edit_mask = board.copy()

    obs = {"board": board, "directions": directions, "edit_mask": edit_mask}

    # 3. Inference
    # deterministic=True makes the AI play its "best" move, not explore
    action, _ = model.predict(obs, deterministic=True)

    # 4. Return new directions
    # The action is a flat array of 100 ints (0-4)
    return jsonify({"new_directions": action.tolist()})


if __name__ == "__main__":
    print("Starting AI Game Server on http://localhost:8000")
    app.run(port=8000, debug=True)
