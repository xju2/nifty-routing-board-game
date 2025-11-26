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
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, placer_extra_pieces=5, placer_strategy=None):
        super(RoutingGameEnv, self).__init__()

        self.W = W
        self.H = H
        self.placer_extra_pieces_total = placer_extra_pieces

        # Pluggable strategy: Default to random if none provided
        self.placer_strategy = (
            placer_strategy if placer_strategy else self._default_random_placer
        )

        # Observation Space
        # 0=Empty, 1=Piece
        # Directions: 1=Up, 2=Right, 3=Down, 4=Left (0 is unused/invalid on board now)
        # Mask: 1=Editable by Router, 0=Locked
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
                "directions": spaces.Box(low=1, high=4, shape=(H, W), dtype=np.uint8),
                "edit_mask": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
                "steps_remaining": spaces.Box(
                    low=0, high=50, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Action Space: 4 options per tile (UP, RIGHT, DOWN, LEFT)
        # We map 0->1, 1->2, 2->3, 3->4 to ensure NO "None" directions.
        self.action_space = spaces.MultiDiscrete([4] * (H * W))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((H, W), dtype=np.uint8)

        # REQUIREMENT: Board initiated with routes.
        # We initialize random directions (1-4) everywhere.
        self.directions = np.random.randint(1, 5, size=(H, W), dtype=np.uint8)

        self.eaten_pieces = 0
        self.steps_in_phase_7 = 0
        self.placer_pieces_left = self.placer_extra_pieces_total

        # --- FAST FORWARD ---
        # Rule: 1. Router sets initial... (We just did this randomly)
        # Rule: 2. Placer places 8 pieces.
        self.placer_strategy(self.board, 8)

        # Rule: 3. Router modifies occupied squares.
        # We start the agent interaction HERE.
        self.edit_mask = self.board.copy()  # Only occupied are editable
        self.phase = 2

        return self._get_obs(), {}

    def _get_obs(self):
        # Approx steps left calculation for agent awareness
        rem = 25.0 if self.placer_pieces_left <= 0 else (25.0 + self.placer_pieces_left)
        return {
            "board": self.board.copy(),
            "directions": self.directions.copy(),
            "edit_mask": self.edit_mask.copy(),
            "steps_remaining": np.array([rem], dtype=np.float32),
        }

    def _apply_router_action(self, action):
        # Reshape flat action to 2D
        action_2d = action.reshape(self.H, self.W)

        # MAP 0-3 to 1-4 (UP, RIGHT, DOWN, LEFT)
        # This ensures every tile always has a valid direction.
        mapped_action = action_2d + 1

        # Update only allowed tiles
        update_indices = np.where(self.edit_mask == 1)
        self.directions[update_indices] = mapped_action[update_indices]

    def _simulation_step(self):
        """Advances board one step."""
        # 1. Clean Output Tile (Start of Turn Logic)
        if self.board[OUT_Y, OUT_X] == 1:
            self.board[OUT_Y, OUT_X] = 0

        new_board = np.zeros((H, W), dtype=np.uint8)
        moves = []

        for y in range(H):
            for x in range(W):
                if self.board[y, x] == 1:
                    d = self.directions[y, x]
                    # d is guaranteed 1-4 now, so no DIR_NONE check needed strictly,
                    # but good to be safe.
                    if d == DIR_NONE:
                        moves.append((y, x, y, x))
                        continue

                    nx, ny = x + DX[d], y + DY[d]

                    # Check Bounds
                    if 0 <= nx < W and 0 <= ny < H:
                        moves.append((y, x, ny, nx))
                    else:
                        # Attempt to move off-board (not at output) -> Stay put
                        moves.append((y, x, y, x))

        # 2. Resolve Collisions
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

        # Note: Output tile logic. Pieces arriving at (OUT_X, OUT_Y) stay
        # visible for 1 frame (so Placer sees them), then removed at start of next.

        self.eaten_pieces += current_eaten
        self.board = new_board

    def _default_random_placer(self, board, count):
        """Randomly places pieces on empty squares."""
        pieces_placed = 0
        attempts = 0
        last_placed = None
        while pieces_placed < count and attempts < 200:
            rx, ry = np.random.randint(0, W), np.random.randint(0, H)
            if board[ry, rx] == 0:
                board[ry, rx] = 1
                pieces_placed += 1
                last_placed = (ry, rx)
            attempts += 1
        return last_placed

    def step(self, action):
        # 1. Apply Agent (Router) Action
        self._apply_router_action(action)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # --- PHASE 2: Main Loop (Modify -> Sim -> Place) ---
        if self.phase == 2:
            # 1. Simulate Step
            self._simulation_step()

            # 2. Placer Turn (if pieces left)
            if self.placer_pieces_left > 0:
                last_pos = self.placer_strategy(self.board, 1)
                self.placer_pieces_left -= 1

                # Setup next mask: New piece + Adjacent
                self.edit_mask = np.zeros((H, W), dtype=np.uint8)
                if last_pos:
                    ly, lx = last_pos
                    shifts = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dy, dx in shifts:
                        ny, nx = ly + dy, lx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            self.edit_mask[ny, nx] = 1

                # Stay in Phase 2

            else:
                # 3. End Game Simulation (Phase 7)
                # Run until empty or step limit
                step_count = 0
                # Limit total run to prevent infinite loops if cycles exist
                while np.sum(self.board) > 0 and step_count < 25:
                    self._simulation_step()
                    step_count += 1

                self.steps_in_phase_7 = step_count

                # Score: Steps + 10 * Eaten + 10 * Leftover
                pieces_left = np.sum(self.board)
                score = (
                    self.steps_in_phase_7
                    + (10 * self.eaten_pieces)
                    + (10 * pieces_left)
                )

                # Reward is negative score
                reward = -float(score)
                terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print("-" * 20)
        print(f"Phase: {self.phase}, Eaten: {self.eaten_pieces}")
        DIR_SYMBOLS = {1: "^", 2: ">", 3: "v", 4: "<"}
        for y in range(H):
            line = ""
            for x in range(W):
                char = "."
                if self.board[y, x] == 1:
                    char = "P"

                d = self.directions[y, x]
                d_char = DIR_SYMBOLS.get(d, "?")

                if char == "P":
                    line += f"[{d_char}]"
                else:
                    line += f" {d_char} "
            print(line)
        print("-" * 20)
