import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Constants from main.c
W, H = 10, 10
OUT_X, OUT_Y = 5, 0

# Directions
DIR_NONE = 0
DIR_UP = 1
DIR_RIGHT = 2
DIR_DOWN = 3
DIR_LEFT = 4

# Mapping for simulation
DX = {DIR_RIGHT: 1, DIR_LEFT: -1, DIR_UP: 0, DIR_DOWN: 0, DIR_NONE: 0}
DY = {DIR_RIGHT: 0, DIR_LEFT: 0, DIR_UP: -1, DIR_DOWN: 1, DIR_NONE: 0}

class RoutingGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, placer_extra_pieces=5):
        super(RoutingGameEnv, self).__init__()

        self.W = W
        self.H = H
        self.placer_extra_pieces_total = placer_extra_pieces

        # 0=Empty, 1=Piece
        # Directions: 0=None, 1=Up, 2=Right, 3=Down, 4=Left
        # Mask: 1=Editable by Router, 0=Locked
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
            "directions": spaces.Box(low=0, high=4, shape=(H, W), dtype=np.uint8),
            "edit_mask": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8)
        })

        # FIX: Flatten the action space to 1D to satisfy Stable Baselines3
        # Instead of (10, 10), we use (100,)
        self.action_space = spaces.MultiDiscrete(np.full(H * W, 5))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((H, W), dtype=np.uint8)
        self.directions = np.zeros((H, W), dtype=np.uint8)
        self.eaten_pieces = 0
        self.steps_in_phase_7 = 0
        self.placer_pieces_left = self.placer_extra_pieces_total

        # Game State Tracker
        # 0: Router sets initial directions (Step 1)
        # 1: Placer places 8 pieces (Step 2) -> Immediate transition to Router Mod 1
        # 2: Router modifies occupied squares (Step 3)
        # 3: Simulation Loop (Step 4-6)
        # 4: Final Simulation (Step 7)
        self.phase = 0

        # In phase 0, Router can edit everything
        self.edit_mask = np.ones((H, W), dtype=np.uint8)

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "directions": self.directions.copy(),
            "edit_mask": self.edit_mask.copy()
        }

    def _apply_router_action(self, action):
        # FIX: Reshape the 1D action back to 2D (H, W) to map to the board
        action_2d = action.reshape(self.H, self.W)

        # Update directions only where mask is 1
        update_indices = np.where(self.edit_mask == 1)
        self.directions[update_indices] = action_2d[update_indices]

    def _simulation_step(self):
        """Advances board one step. Handles movement and collisions."""
        new_board = np.zeros((H, W), dtype=np.uint8)
        moves = [] # List of (y, x, target_y, target_x)

        # 1. Calculate moves
        for y in range(H):
            for x in range(W):
                if self.board[y, x] == 1:
                    d = self.directions[y, x]
                    if d == DIR_NONE:
                        moves.append((y, x, y, x))
                        continue

                    nx, ny = x + DX[d], y + DY[d]

                    # Check Bounds
                    if 0 <= nx < W and 0 <= ny < H:
                        moves.append((y, x, ny, nx))
                    else:
                        moves.append((y, x, y, x))


        # 2. Execute moves & Collision
        counts = np.zeros((H, W), dtype=int)
        for _, _, ty, tx in moves:
            counts[ty, tx] += 1

        current_eaten = 0

        for ty in range(H):
            for tx in range(W):
                c = counts[ty, tx]
                if c > 0:
                    if c > 1:
                        current_eaten += (c - 1)
                    new_board[ty, tx] = 1

        # 3. Clean Output Tile
        if new_board[OUT_Y, OUT_X] == 1:
            new_board[OUT_Y, OUT_X] = 0

        self.eaten_pieces += current_eaten
        self.board = new_board

    def _placer_action_random(self, count=1):
        """Randomly places `count` pieces on empty squares."""
        pieces_placed = 0
        attempts = 0
        last_placed = None
        while pieces_placed < count and attempts < 100:
            rx, ry = np.random.randint(0, W), np.random.randint(0, H)
            if self.board[ry, rx] == 0:
                self.board[ry, rx] = 1
                pieces_placed += 1
                last_placed = (ry, rx)
            attempts += 1
        return last_placed

    def step(self, action):
        # 1. Apply Agent (Router) Action based on current mask
        self._apply_router_action(action)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # ------------------------------------------------
        # PHASE LOGIC
        # ------------------------------------------------

        if self.phase == 0:
            # Step 1 Complete -> Step 2 (Place 8)
            self._placer_action_random(8)
            self.edit_mask = self.board.copy()
            self.phase = 2
            return self._get_obs(), reward, terminated, truncated, info

        elif self.phase == 2:
            # Step 3/6 Complete -> Step 4 (Simulate)
            self._simulation_step()

            # Step 5 (Place 1)
            if self.placer_pieces_left > 0:
                last_pos = self._placer_action_random(1)
                self.placer_pieces_left -= 1

                # Step 6 Setup (Edit new + adjacent)
                self.edit_mask = np.zeros((H, W), dtype=np.uint8)
                if last_pos:
                    ly, lx = last_pos
                    shifts = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
                    for dy, dx in shifts:
                        ny, nx = ly + dy, lx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            self.edit_mask[ny, nx] = 1

                # Remain in Phase 2 logic (Router edit -> Sim loop)
                self.phase = 2

            else:
                # Step 7: Final Simulation Run
                step_count = 0
                while np.sum(self.board) > 0 and step_count < 25:
                    self._simulation_step()
                    step_count += 1

                self.steps_in_phase_7 = step_count

                # Calculate Score
                pieces_left = np.sum(self.board)
                score = self.steps_in_phase_7 + (10 * self.eaten_pieces) + (10 * pieces_left)

                reward = -float(score)
                terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print("-" * 20)
        print(f"Phase: {self.phase}, Eaten: {self.eaten_pieces}")
        for y in range(H):
            line = ""
            for x in range(W):
                char = "."
                if self.board[y, x] == 1:
                    char = "P"

                d = self.directions[y, x]
                d_char = " "
                if d == DIR_UP: d_char = "^"
                elif d == DIR_RIGHT: d_char = ">"
                elif d == DIR_DOWN: d_char = "v"
                elif d == DIR_LEFT: d_char = "<"

                if char == "P":
                    line += f"[{d_char}]"
                else:
                    line += f" {d_char} "
            print(line)
        print("-" * 20)