import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import os

# ==========================================
# 1. Interactive Environment Definition
# ==========================================

# Constants
W, H = 10, 10
OUT_X, OUT_Y = 5, 0

# Directions
DIR_NONE = 0
DIR_UP = 1
DIR_RIGHT = 2
DIR_DOWN = 3
DIR_LEFT = 4
DIR_CHARS = {DIR_NONE: " ", DIR_UP: "^", DIR_RIGHT: ">", DIR_DOWN: "v", DIR_LEFT: "<"}

DX = {DIR_RIGHT: 1, DIR_LEFT: -1, DIR_UP: 0, DIR_DOWN: 0, DIR_NONE: 0}
DY = {DIR_RIGHT: 0, DIR_LEFT: 0, DIR_UP: -1, DIR_DOWN: 1, DIR_NONE: 0}


class InteractiveRoutingGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, placer_extra_pieces=5):
        super(InteractiveRoutingGameEnv, self).__init__()
        self.W = W
        self.H = H
        self.placer_extra_pieces_total = placer_extra_pieces

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
                "directions": spaces.Box(low=0, high=4, shape=(H, W), dtype=np.uint8),
                "edit_mask": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
            }
        )

        # Flat action space (same as training)
        self.action_space = spaces.MultiDiscrete([5] * (H * W))
        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros((H, W), dtype=np.uint8)
        self.directions = np.zeros((H, W), dtype=np.uint8)
        self.eaten_pieces = 0
        self.steps_in_phase_7 = 0
        self.placer_pieces_left = self.placer_extra_pieces_total
        self.phase = 0
        self.edit_mask = np.ones((H, W), dtype=np.uint8)
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "directions": self.directions.copy(),
            "edit_mask": self.edit_mask.copy(),
        }

    def _apply_router_action(self, action):
        action_2d = action.reshape(self.H, self.W)
        update_indices = np.where(self.edit_mask == 1)
        self.directions[update_indices] = action_2d[update_indices]

    def _simulation_step(self):
        print("\n--- Simulating Step ---")
        new_board = np.zeros((H, W), dtype=np.uint8)
        moves = []

        for y in range(H):
            for x in range(W):
                if self.board[y, x] == 1:
                    d = self.directions[y, x]
                    if d == DIR_NONE:
                        moves.append((y, x, y, x))
                        continue
                    nx, ny = x + DX[d], y + DY[d]
                    if 0 <= nx < W and 0 <= ny < H:
                        moves.append((y, x, ny, nx))
                    else:
                        moves.append((y, x, y, x))

        counts = np.zeros((H, W), dtype=int)
        for _, _, ty, tx in moves:
            counts[ty, tx] += 1

        current_eaten = 0
        for ty in range(H):
            for tx in range(W):
                c = counts[ty, tx]
                if c > 0:
                    if c > 1:
                        current_eaten += c - 1
                    new_board[ty, tx] = 1

        # Output logic
        if new_board[OUT_Y, OUT_X] == 1:
            print(f"Piece exited at ({OUT_X}, {OUT_Y})!")
            new_board[OUT_Y, OUT_X] = 0

        if current_eaten > 0:
            print(f"CRASH! {current_eaten} piece(s) eaten this step.")

        self.eaten_pieces += current_eaten
        self.board = new_board
        self.render()

    def _get_user_input(self, prompt):
        while True:
            try:
                user_in = input(prompt)
                parts = user_in.split()
                if len(parts) != 2:
                    print("Please enter exactly two numbers: x y")
                    continue
                x, y = int(parts[0]), int(parts[1])
                if 0 <= x < W and 0 <= y < H:
                    if self.board[y, x] == 0:
                        return x, y
                    else:
                        print("That tile is already occupied!")
                else:
                    print(f"Coordinates out of bounds (0-{W-1}, 0-{H-1})")
            except ValueError:
                print("Invalid input. Please enter numbers.")

    def _placer_action_human(self, count=1):
        self.render()
        print(f"\n[YOUR TURN] You need to place {count} piece(s).")
        last_placed = None
        for i in range(count):
            print(f"Piece {i+1}/{count}:")
            x, y = self._get_user_input("Enter coordinates (x y): ")
            self.board[y, x] = 1
            last_placed = (y, x)
            self.render()
        return last_placed

    def step(self, action):
        # 1. AI (Router) acts first
        print("\n[AI Turn] Router is updating arrows...")
        self._apply_router_action(action)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.phase == 0:
            # Step 1 Done. Now Step 2: Human places 8 pieces
            self._placer_action_human(8)
            self.edit_mask = self.board.copy()
            self.phase = 2
            return self._get_obs(), reward, terminated, truncated, info

        elif self.phase == 2:
            # Step 3/6 Done. Step 4: Simulation
            self._simulation_step()

            # Step 5: Human places 1 piece (if any left)
            if self.placer_pieces_left > 0:
                last_pos = self._placer_action_human(1)
                self.placer_pieces_left -= 1

                # Setup mask for Step 6
                self.edit_mask = np.zeros((H, W), dtype=np.uint8)
                if last_pos:
                    ly, lx = last_pos
                    shifts = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dy, dx in shifts:
                        ny, nx = ly + dy, lx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            self.edit_mask[ny, nx] = 1
                self.phase = 2
            else:
                # Step 7: Final Simulation Run
                print(
                    "\n[Endgame] No pieces left to place. Simulating remaining steps..."
                )
                step_count = 0
                while np.sum(self.board) > 0 and step_count < 25:
                    self._simulation_step()
                    step_count += 1
                    # input("Press Enter to advance step...") # Optional: Un-comment to step slowly

                self.steps_in_phase_7 = step_count
                pieces_left = np.sum(self.board)
                score = (
                    self.steps_in_phase_7
                    + (10 * self.eaten_pieces)
                    + (10 * pieces_left)
                )

                print(f"\nGAME OVER! Final Score: {score} (Lower is better)")
                print(
                    f"Steps: {self.steps_in_phase_7}, Eaten: {self.eaten_pieces}, Left: {pieces_left}"
                )
                terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print("\n   " + "".join([f"{i}  " for i in range(W)]))
        print("  " + "-" * (W * 3))
        for y in range(H):
            line = f"{y}| "
            for x in range(W):
                char = "."
                if self.board[y, x] == 1:
                    char = "P"  # Piece

                d = self.directions[y, x]
                d_char = DIR_CHARS.get(d, " ")

                if char == "P":
                    line += f"\033[91m{d_char}\033[0m  "  # Red P-like (actually showing arrow on piece)
                else:
                    line += f"{d_char}  "
            print(line)
        print("  " + "-" * (W * 3))


# ==========================================
# 2. Main Game Loop
# ==========================================


def play_game(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Could not find model file '{model_path}'.")
        print("Make sure you trained the model and the file is in this directory.")
        exit()

    print("Loading AI Model...")
    model = PPO.load(model_path)

    print("\nStarting Interactive Game!")
    print("You are the Placer. The AI is the Router.")
    print("Goal: Force the AI to get a HIGH score (Cause collisions/delays).")
    print("AI Goal: Get a LOW score (Route pieces to top output (5,0)).")

    env = InteractiveRoutingGameEnv(placer_extra_pieces=5)
    obs, _ = env.reset()

    terminated = False
    while not terminated:
        # AI Predicts Action
        action, _ = model.predict(obs, deterministic=True)

        # Environment steps (This will trigger input prompts for you)
        obs, reward, terminated, truncated, info = env.step(action)
