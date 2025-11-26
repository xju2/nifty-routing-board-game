import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Constants
W, H = 10, 10
OUT_X, OUT_Y = 5, 0

# Directions (1-4 Only, 0 is unused/invalid for routing)
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

    def __init__(self, placer_extra_pieces=5):
        super(RoutingGameEnv, self).__init__()

        self.W = W
        self.H = H
        self.placer_extra_pieces_total = placer_extra_pieces

        # Observation Space
        # board: 0=Empty, 1=Piece
        # directions: 1=Up, 2=Right, 3=Down, 4=Left
        # edit_mask: Now always 1s (Full control)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
                "directions": spaces.Box(low=1, high=4, shape=(H, W), dtype=np.uint8),
                "edit_mask": spaces.Box(low=0, high=1, shape=(H, W), dtype=np.uint8),
            }
        )

        # Action Space: 4 options per tile (UP, RIGHT, DOWN, LEFT)
        # We map 0->1, 1->2, 2->3, 3->4 to ensure NO "None" directions.
        # This enforces "every box should have a routing direction".
        self.action_space = spaces.MultiDiscrete([4] * (H * W))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((H, W), dtype=np.uint8)

        # 1. Initialize board with RANDOM valid routes (1-4)
        self.directions = np.random.randint(1, 5, size=(H, W), dtype=np.uint8)

        self.eaten_pieces = 0
        self.steps_in_phase_7 = 0
        self.placer_pieces_left = self.placer_extra_pieces_total

        # 2. Placer places the initial 8 pieces randomly
        self._placer_action_random(8)

        # 3. Router Turn: Agent can now edit the WHOLE board
        self.edit_mask = np.ones((H, W), dtype=np.uint8)
        self.phase = 2

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "directions": self.directions.copy(),
            "edit_mask": self.edit_mask.copy(),
        }

    def _apply_router_action(self, action):
        # Reshape flat action (100,) to 2D (10, 10)
        action_2d = action.reshape(self.H, self.W)

        # Map Agent Output (0-3) to Game Directions (1-4)
        # 0->UP(1), 1->RIGHT(2), 2->DOWN(3), 3->LEFT(4)
        mapped_action = action_2d + 1

        # Update all tiles (since mask is all 1s)
        # We still use the mask logic just in case you want to restrict it later
        update_indices = np.where(self.edit_mask == 1)
        self.directions[update_indices] = mapped_action[update_indices]

    def _simulation_step(self):
        """Advances board one step."""
        # 1. Clean Output Tile (Start of Turn Rule)
        if self.board[OUT_Y, OUT_X] == 1:
            self.board[OUT_Y, OUT_X] = 0

        new_board = np.zeros((H, W), dtype=np.uint8)
        moves = []

        for y in range(H):
            for x in range(W):
                if self.board[y, x] == 1:
                    d = self.directions[y, x]
                    # Since we force d in [1..4], DIR_NONE(0) is impossible
                    if d == DIR_NONE:
                        moves.append((y, x, y, x))
                        continue

                    nx, ny = x + DX[d], y + DY[d]

                    # Check Bounds
                    if 0 <= nx < W and 0 <= ny < H:
                        moves.append((y, x, ny, nx))
                    else:
                        # Moves off board (not at output) -> Stay put
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

        self.eaten_pieces += current_eaten
        self.board = new_board

    def _placer_action_random(self, count=1):
        """Randomly places `count` pieces on empty squares."""
        pieces_placed = 0
        attempts = 0
        last_placed = None
        while pieces_placed < count and attempts < 200:
            rx, ry = np.random.randint(0, W), np.random.randint(0, H)
            if self.board[ry, rx] == 0:
                self.board[ry, rx] = 1
                pieces_placed += 1
                last_placed = (ry, rx)
            attempts += 1
        return last_placed

    def step(self, action):
        # 1. Apply Agent (Router) Action
        # This will now update the ENTIRE board because edit_mask is all 1s
        self._apply_router_action(action)

        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.phase == 2:
            # 2. Simulate Step
            self._simulation_step()

            # 3. Placer Turn
            if self.placer_pieces_left > 0:
                self._placer_action_random(1)
                self.placer_pieces_left -= 1

                # Reset Mask to Full Board (Agent can fix routing anywhere)
                self.edit_mask = np.ones((H, W), dtype=np.uint8)

                # Continue loop (Phase 2)

            else:
                # 4. End Game Simulation (Phase 7)
                step_count = 0
                while np.sum(self.board) > 0 and step_count < 25:
                    self._simulation_step()
                    step_count += 1

                self.steps_in_phase_7 = step_count

                # Score Calculation
                pieces_left = np.sum(self.board)
                score = (
                    self.steps_in_phase_7
                    + (10 * self.eaten_pieces)
                    + (10 * pieces_left)
                )

                # Negative score for reward (Minimize score)
                reward = -float(score)
                terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print("-" * 20)
        print(f"Phase: {self.phase}, Eaten: {self.eaten_pieces}")
        # 1=^, 2=>, 3=v, 4=<
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
