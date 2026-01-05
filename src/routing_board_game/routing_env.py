"""
Gymnasium RL environment for grid-based routing board game.

The AI agent controls multiple pieces and moves them one at a time during its turn
toward a root square, while the user can introduce additional pieces that
immediately join the AI-controlled swarm and route toward the root.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, List

from routing_board_game.policies import BaseMovePolicy, GreedyPolicy


class RoutingBoardGameEnv(gym.Env):
    """
    Gymnasium environment for routing board game with multi-move AI routing phase.

    Board:
        - 10×10 grid with (0,0) at top-left
        - Root square at (0, 5) - unlimited capacity

    AI Pieces:
        - Default: 8 pieces (configurable)
        - Move toward root reducing Manhattan distance
        - Each piece moves once per turn

    Blocking Pieces:
        - User-controlled (placed by action)
        - Immediately convert into AI-controlled pieces that also route to the root

    Turn Structure (one env.step):
        1. User placement phase: place one blocking piece (becomes AI piece)
        2. AI routing phase: move all AI pieces sequentially

    Rewards:
        - -1 per routing phase
        - +1 when a piece reaches root
        - Optional: shaping based on distance reduction

    Termination:
        - All AI pieces reach root (success)
        - No AI piece has legal forward move (failure)
        - Max steps reached (default: 10)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_ai_pieces: int = 8,
        max_steps: int = 10,
        reward_shaping: bool = True,
        render_mode: Optional[str] = None,
        move_policy: Optional[BaseMovePolicy] = None,
    ):
        super().__init__()

        self.grid_size = 10
        self.root_pos = (0, 5)  # (row, col)
        self.num_ai_pieces = num_ai_pieces
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        self.move_policy: BaseMovePolicy = move_policy or GreedyPolicy()

        # Movement directions: up, down, left, right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Action space: place blocking piece at (row, col) that becomes an AI piece
        # We use a flat index: action = row * 10 + col
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Observation space: board with piece types
        # 0 = empty, 1 = AI piece (including user-added), 3 = root
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

        # State variables
        self.board = None
        self.ai_pieces = None  # List of (row, col) positions
        self.step_count = 0
        self.pieces_routed = 0
        self.user_pieces_added = 0
        self.pieces_spawned = 0  # Tracks original AI pieces + user-added pieces

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize empty board
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Mark root square
        self.board[self.root_pos] = 3

        # Randomly place AI pieces (not on root)
        self.ai_pieces = []
        self.user_pieces_added = 0

        occupied = {self.root_pos}
        attempts = 0
        while len(self.ai_pieces) < self.num_ai_pieces and attempts < 1000:
            row = self.np_random.integers(0, self.grid_size)
            col = self.np_random.integers(0, self.grid_size)
            pos = (row, col)

            if pos not in occupied:
                self.ai_pieces.append(pos)
                occupied.add(pos)
                self.board[pos] = 1
            attempts += 1

        if len(self.ai_pieces) < self.num_ai_pieces:
            raise RuntimeError(f"Failed to place all {self.num_ai_pieces} AI pieces")

        self.step_count = 0
        self.pieces_routed = 0
        self.pieces_spawned = len(self.ai_pieces)

        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        """Get current observation."""
        return {
            "board": self.board.copy(),
            "step_count": np.array([self.step_count], dtype=np.int32),
        }

    def _manhattan_distance(self, pos: Tuple[int, int]) -> int:
        """Calculate Manhattan distance from position to root."""
        return abs(pos[0] - self.root_pos[0]) + abs(pos[1] - self.root_pos[1])

    def _get_forward_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all legal forward moves for a piece at given position.
        Forward move = reduces Manhattan distance to root.
        """
        current_dist = self._manhattan_distance(pos)
        forward_moves = []

        for dr, dc in self.directions:
            new_row = pos[0] + dr
            new_col = pos[1] + dc
            new_pos = (new_row, new_col)

            # Check bounds
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                continue

            # Check if move reduces distance
            new_dist = self._manhattan_distance(new_pos)
            if new_dist < current_dist:
                # Root can hold unlimited pieces
                if new_pos == self.root_pos:
                    forward_moves.append(new_pos)
                elif self.board[new_pos] == 0:
                    # Only move into empty squares; board is updated as pieces move
                    forward_moves.append(new_pos)

        return forward_moves

    def _is_piece_blocked(self, pos: Tuple[int, int]) -> bool:
        """Check if a piece at given position is blocked (no legal forward moves)."""
        return len(self._get_forward_moves(pos)) == 0

    def _place_blocking_piece(self, action: int) -> bool:
        """
        Place a user-controlled piece that immediately becomes AI-controlled.

        Returns True if placement was successful, False otherwise. A successful
        placement increases the total AI pieces that will attempt to route.
        """
        row = action // self.grid_size
        col = action % self.grid_size
        pos = (row, col)

        # Cannot place on root
        if pos == self.root_pos:
            return False

        # Cannot place on occupied square
        if pos in self.ai_pieces:
            return False

        # Place new AI piece
        self.ai_pieces.append(pos)
        self.board[pos] = 1
        self.user_pieces_added += 1
        self.pieces_spawned += 1
        return True

    def _execute_ai_routing_phase(
        self,
    ) -> Tuple[float, int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Execute the AI routing phase: move all AI pieces once each.

        Returns:
            reward: reward for this routing phase
            pieces_reached_root: number of pieces that reached root
            move_sequence: list of (from_pos, to_pos) tuples for visualization
        """
        total_distance_before = sum(
            self._manhattan_distance(pos) for pos in self.ai_pieces
        )
        pieces_reached_root = 0
        move_sequence: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # Track which pieces have been moved this turn
        pieces_to_move = list(self.ai_pieces)
        moved_pieces = []

        # Keep moving pieces until all have moved or are blocked
        while pieces_to_move:
            # Find a piece that can move
            moved_this_iteration = False

            for i, pos in enumerate(pieces_to_move):
                forward_moves = self._get_forward_moves(pos)

                if forward_moves:
                    src_pos = pos
                    new_pos = self.move_policy.select_move(
                        piece_pos=pos,
                        forward_moves=forward_moves,
                        board=self.board,
                        root_pos=self.root_pos,
                        rng=self.np_random,
                    )
                    # Fallback safety: if policy returns invalid move, pick randomly
                    if new_pos not in forward_moves:
                        new_pos = forward_moves[
                            self.np_random.integers(0, len(forward_moves))
                        ]

                    # Clear old position on board
                    if pos != self.root_pos:
                        self.board[pos] = 0

                    # Move piece
                    pieces_to_move.pop(i)

                    # Check if reached root
                    if new_pos == self.root_pos:
                        pieces_reached_root += 1
                        # Don't add to moved_pieces (piece is removed from game)
                        # Root is marked on board, don't change it
                    else:
                        moved_pieces.append(new_pos)
                        self.board[new_pos] = 1
                    move_sequence.append((src_pos, new_pos))

                    moved_this_iteration = True
                    break

            # If no piece could move in this iteration, remaining pieces are blocked
            if not moved_this_iteration:
                # All remaining pieces are blocked
                moved_pieces.extend(pieces_to_move)
                break

        # Update AI pieces list (excluding routed pieces)
        self.ai_pieces = moved_pieces

        # Calculate reward
        reward = -1.0  # Base cost per routing phase
        reward += pieces_reached_root  # +1 per piece reaching root

        # Optional reward shaping based on distance reduction
        if self.reward_shaping and self.ai_pieces:
            total_distance_after = sum(
                self._manhattan_distance(pos) for pos in self.ai_pieces
            )
            distance_reduction = (
                total_distance_before - total_distance_after - pieces_reached_root
            )
            reward += 0.1 * distance_reduction  # Small bonus for progress

        return reward, pieces_reached_root, move_sequence

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: Index for placing a user piece (0-99 for 10x10 grid)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Phase 1: User places a piece that will route toward the root
        placement_success = self._place_blocking_piece(action)

        # Phase 2: AI routing phase (move all pieces)
        reward, pieces_reached_root, move_sequence = self._execute_ai_routing_phase()

        self.step_count += 1
        self.pieces_routed += pieces_reached_root

        # Check termination conditions
        terminated = False
        truncated = False

        # Success: all AI pieces reached root
        if len(self.ai_pieces) == 0:
            terminated = True

        # Failure: no AI piece has a legal forward move
        elif all(self._is_piece_blocked(pos) for pos in self.ai_pieces):
            terminated = True
            reward -= 10.0  # Penalty for failure

        # Max steps reached
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "placement_success": placement_success,
            "pieces_reached_root": pieces_reached_root,
            "pieces_remaining": len(self.ai_pieces),
            "pieces_routed_total": self.pieces_routed,
            "pieces_spawned_total": self.pieces_spawned,
            "user_pieces_added": self.user_pieces_added,
            "step_count": self.step_count,
            "move_sequence": move_sequence,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Render the current state of the environment."""
        if self.render_mode is None:
            return

        print("\n" + "=" * 40)
        print(f"Step: {self.step_count}/{self.max_steps}")
        print(f"AI Pieces: {len(self.ai_pieces)} | Routed: {self.pieces_routed}")
        print("=" * 40)

        # Print column headers
        print("   ", end="")
        for col in range(self.grid_size):
            print(f"{col:2}", end=" ")
        print()

        # Print board
        for row in range(self.grid_size):
            print(f"{row:2} ", end="")
            for col in range(self.grid_size):
                pos = (row, col)
                if pos == self.root_pos:
                    print(" R", end=" ")  # Root
                elif pos in self.ai_pieces:
                    print(" A", end=" ")  # AI piece
                else:
                    print(" .", end=" ")  # Empty
            print()
        print("=" * 40)
        print("Legend: R=Root, A=AI piece (including user-added), .=Empty\n")
