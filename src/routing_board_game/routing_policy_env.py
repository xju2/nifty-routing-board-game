"""
PPO-friendly environment where the agent controls routing decisions directly.

Action space: MultiDiscrete([100, 100]) representing (src_index, dst_index)
 - src_index = row * 10 + col of the piece to move
 - dst_index = row * 10 + col of the desired destination

If the move is invalid (no piece at src, dst not a legal forward move), the
step applies a small penalty and no movement occurs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RoutingPolicyEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_ai_pieces: int = 8,
        max_steps: int = 200,
        reward_shaping: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.grid_size = 10
        self.root_pos = (0, 5)
        self.num_ai_pieces = num_ai_pieces
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete(
            [self.grid_size * self.grid_size, self.grid_size * self.grid_size]
        )
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.grid_size, self.grid_size),
                    dtype=np.uint8,
                ),
                "step_count": spaces.Box(
                    low=0, high=max_steps, shape=(1,), dtype=np.int32
                ),
            }
        )

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.board: np.ndarray = None
        self.ai_pieces: List[Tuple[int, int]] = []
        self.step_count = 0
        self.pieces_routed = 0

    def _manhattan_distance(self, pos: Tuple[int, int]) -> int:
        return abs(pos[0] - self.root_pos[0]) + abs(pos[1] - self.root_pos[1])

    def _get_forward_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        current_dist = self._manhattan_distance(pos)
        moves = []
        for dr, dc in self.directions:
            r, c = pos[0] + dr, pos[1] + dc
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                continue
            new_pos = (r, c)
            new_dist = self._manhattan_distance(new_pos)
            if new_dist < current_dist:
                if new_pos == self.root_pos or self.board[new_pos] == 0:
                    moves.append(new_pos)
        return moves

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.board[self.root_pos] = 3
        self.ai_pieces = []
        occupied = {self.root_pos}
        attempts = 0
        while len(self.ai_pieces) < self.num_ai_pieces and attempts < 1000:
            row = self.np_random.integers(0, self.grid_size)
            col = self.np_random.integers(0, self.grid_size)
            pos = (row, col)
            if pos not in occupied:
                self.ai_pieces.append(pos)
                self.board[pos] = 1
                occupied.add(pos)
            attempts += 1
        if len(self.ai_pieces) < self.num_ai_pieces:
            raise RuntimeError("Failed to place all AI pieces")
        self.step_count = 0
        self.pieces_routed = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "step_count": np.array([self.step_count], dtype=np.int32),
        }

    def step(self, action: Tuple[int, int]):
        src_idx, dst_idx = int(action[0]), int(action[1])
        src = (src_idx // self.grid_size, src_idx % self.grid_size)
        dst = (dst_idx // self.grid_size, dst_idx % self.grid_size)

        reward = -0.01  # small per-move cost
        terminated = False
        truncated = False
        move_sequence = []
        placement_success = False

        if src in self.ai_pieces:
            forward_moves = self._get_forward_moves(src)
            if dst in forward_moves:
                placement_success = True
                # apply move
                if src != self.root_pos:
                    self.board[src] = 0
                self.ai_pieces.remove(src)
                if dst == self.root_pos:
                    reward += 1.0
                    self.pieces_routed += 1
                else:
                    self.ai_pieces.append(dst)
                    self.board[dst] = 1
                    # reward shaping: distance reduction
                    if self.reward_shaping:
                        reward += 0.1 * (
                            self._manhattan_distance(src)
                            - self._manhattan_distance(dst)
                        )
                move_sequence.append((src, dst))
            else:
                reward -= 0.1  # invalid dst
        else:
            reward -= 0.1  # invalid src

        self.step_count += 1

        if len(self.ai_pieces) == 0:
            terminated = True
        elif all(len(self._get_forward_moves(pos)) == 0 for pos in self.ai_pieces):
            terminated = True
            reward -= 5.0

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "placement_success": placement_success,
            "pieces_reached_root": self.pieces_routed,
            "pieces_remaining": len(self.ai_pieces),
            "pieces_routed_total": self.pieces_routed,
            "pieces_spawned_total": self.num_ai_pieces,
            "user_pieces_added": 0,
            "step_count": self.step_count,
            "move_sequence": move_sequence,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        print("\n" + "=" * 40)
        print(f"Step: {self.step_count}/{self.max_steps}")
        print(f"AI Pieces: {len(self.ai_pieces)} | Routed: {self.pieces_routed}")
        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                pos = (row, col)
                if pos == self.root_pos:
                    ch = "R"
                elif pos in self.ai_pieces:
                    ch = "A"
                else:
                    ch = "."
                line += ch + " "
            print(line)
        print("=" * 40)
