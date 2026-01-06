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

from routing_board_game.policies import BaseRoutingPolicy, GreedyRoutingPolicy


class RoutingBoardGameEnv(gym.Env):
    """
    Gymnasium environment for routing board game with multi-move AI routing phase.

        Board:
            - 10×10 grid with (0,0) at top-left
            - Root square at (0, 5) - unlimited capacity

        AI Pieces:
            - Start empty; user may place up to 100 pieces (one per turn)
            - Move toward root reducing Manhattan distance
            - Each piece moves once per turn; all pieces must move

        Blocking/User Pieces:
            - User-controlled (placed by action)
            - Immediately convert into AI-controlled pieces that also route to the root

        Turn Structure (one env.step):
            1. User placement phase: place one blocking piece (becomes AI piece)
            2. AI routing phase: move all AI pieces sequentially

        Rewards:
            - -1 per turn
            - Losing a piece: -(5 + Manhattan distance to root)
            - Success bonus when board is cleared (default: +50)
            - Optional: shaping based on distance reduction

        Termination:
            - No AI pieces remain on board
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_ai_pieces: int = 0,
        max_steps: int = 150,
        reward_shaping: bool = False,
        render_mode: Optional[str] = None,
        move_policy: Optional[BaseRoutingPolicy] = None,
        success_bonus: float = 50.0,
    ):
        super().__init__()

        self.grid_size = 10
        self.root_pos = (0, 5)  # (row, col)
        self.num_ai_pieces = num_ai_pieces
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        self.success_bonus = success_bonus
        self.move_policy: BaseRoutingPolicy = move_policy or GreedyRoutingPolicy()

        # Movement directions: up, down, left, right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Action space: place user piece at (row, col) that becomes an AI piece
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
                "user_pieces_remaining": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
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
        self.user_pieces_remaining = 100

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize empty board
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Mark root square
        self.board[self.root_pos] = 3

        # Start with no AI pieces; user must place them
        self.ai_pieces = []
        self.user_pieces_added = 0
        self.step_count = 0
        self.pieces_routed = 0
        self.pieces_spawned = 0
        self.user_pieces_remaining = 100

        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        """Get current observation."""
        return {
            "board": self.board.copy(),
            "step_count": np.array([self.step_count], dtype=np.int32),
            "user_pieces_remaining": np.array(
                [self.user_pieces_remaining], dtype=np.int32
            ),
        }

    def _manhattan_distance(self, pos: Tuple[int, int]) -> int:
        """Calculate Manhattan distance from position to root."""
        return abs(pos[0] - self.root_pos[0]) + abs(pos[1] - self.root_pos[1])

    def _get_forward_moves(
        self, pos: Tuple[int, int], board: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int]]:
        """
        Get all legal forward moves for a piece at given position.
        Forward move = reduces Manhattan distance to root.
        """
        board_ref = board if board is not None else self.board
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
                elif board_ref[new_pos] == 0:
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
        if self.user_pieces_remaining <= 0:
            return False
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
        self.user_pieces_remaining -= 1
        return True

    def _execute_ai_routing_phase(
        self,
    ) -> Tuple[float, int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Execute the AI routing phase using the configured routing policy.

        Returns:
            reward: reward for this routing phase
            pieces_reached_root: number of pieces that reached root
            move_sequence: list of (from_pos, to_pos) tuples for visualization
        """
        total_distance_before = sum(
            self._manhattan_distance(pos) for pos in self.ai_pieces
        )
        pieces_reached_root = 0
        eaten_distances: List[int] = []
        move_sequence: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # Ask policy to propose a routing plan for this turn
        proposed_moves = self.move_policy.plan_moves(
            ai_pieces=list(self.ai_pieces),
            board=self.board.copy(),
            root_pos=self.root_pos,
            forward_move_fn=self._get_forward_moves,
            rng=self.np_random,
        )

        # Apply proposed moves sequentially with safety checks and collision rules
        current_positions = list(self.ai_pieces)
        destination_map: dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for src_pos, new_pos in proposed_moves:
            if src_pos not in current_positions:
                continue  # source no longer has a piece

            forward_moves = self._get_forward_moves(src_pos)
            if new_pos not in forward_moves:
                # invalid move -> piece lost
                eaten_distances.append(self._manhattan_distance(src_pos))
                if src_pos != self.root_pos:
                    self.board[src_pos] = 0
                current_positions.remove(src_pos)
                continue

            # Clear old position
            if src_pos != self.root_pos:
                self.board[src_pos] = 0
            current_positions.remove(src_pos)

            destination_map.setdefault(new_pos, []).append(src_pos)
            move_sequence.append((src_pos, new_pos))

        # Any piece not moved is eaten (must move rule)
        for leftover in current_positions:
            eaten_distances.append(self._manhattan_distance(leftover))
            if leftover != self.root_pos:
                self.board[leftover] = 0

        # Resolve destinations
        next_positions = []
        for dest, src_list in destination_map.items():
            if dest == self.root_pos:
                # Only one survives on root; extras are eaten
                pieces_reached_root += 1
                if len(src_list) > 1:
                    for src in src_list[1:]:
                        eaten_distances.append(self._manhattan_distance(src))
            else:
                # Only one survives per square; extras eaten
                next_positions.append(dest)
                if len(src_list) > 1:
                    for src in src_list[1:]:
                        eaten_distances.append(self._manhattan_distance(src))
                self.board[dest] = 1

        self.ai_pieces = next_positions

        # Calculate reward
        reward = -1.0  # Base cost per routing phase (turn)
        if eaten_distances:
            reward -= sum(5 + d for d in eaten_distances)

        # No additional reward for reaching root under new spec

        # Optional reward shaping based on distance reduction
        if self.reward_shaping and self.ai_pieces:
            total_distance_after = sum(
                self._manhattan_distance(pos) for pos in self.ai_pieces
            )
            distance_reduction = total_distance_before - total_distance_after
            reward += 0.1 * distance_reduction  # Small bonus for progress

        return reward, pieces_reached_root, move_sequence, len(eaten_distances)

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
        reward, pieces_reached_root, move_sequence, eaten_pieces = (
            self._execute_ai_routing_phase()
        )

        self.step_count += 1
        self.pieces_routed += pieces_reached_root

        # Check termination conditions
        terminated = False
        truncated = False

        # Termination: no pieces remain
        if len(self.ai_pieces) == 0:
            reward += self.success_bonus
            terminated = True

        info = {
            "placement_success": placement_success,
            "pieces_reached_root": pieces_reached_root,
            "pieces_remaining": len(self.ai_pieces),
            "pieces_routed_total": self.pieces_routed,
            "pieces_spawned_total": self.pieces_spawned,
            "user_pieces_added": self.user_pieces_added,
            "step_count": self.step_count,
            "move_sequence": move_sequence,
            "eaten_pieces": eaten_pieces,
            "user_pieces_remaining": self.user_pieces_remaining,
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
