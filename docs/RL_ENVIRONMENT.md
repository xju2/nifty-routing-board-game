# Routing Board Game - RL Environment

This directory contains a Gymnasium-compatible RL environment for a grid-based routing board game where the user can add new pieces each turn that immediately become AI-controlled and route toward a root square.

## Game Overview

### Board
- 10×10 grid with (0,0) at top-left
- Root square at position (0, 5) - can hold unlimited pieces

### Pieces
- **AI Pieces**: 8 pieces (configurable) that try to reach the root
- **User Pieces**: User-controlled pieces placed each turn; once placed they become AI-controlled movers toward the root

### Turn Structure
Each `env.step(action)` executes:
1. **User Placement Phase**: Place one piece on the board (it immediately becomes an AI piece)
2. **AI Routing Phase**: Move all AI pieces sequentially toward root
   - Each piece moves once per turn
   - Pieces move 4-directionally (up/down/left/right)
   - Only "forward moves" allowed (moves that reduce Manhattan distance)
   - Dynamic blocking: pieces can become unblocked as others move

### Movement Rules
- Moves are one square in 4 directions (no diagonals)
- Must stay in bounds
- Must move into unoccupied square
- "Forward move" = strictly reduces Manhattan distance to root
- Blocking status is recomputed after every move

### Rewards
- `-1` per routing phase (step)
- `+1` when a piece reaches root
- Optional reward shaping for distance reduction

### Termination
Episode ends when:
- All AI pieces reach root (success)
- No AI piece has legal forward move (failure, -10 penalty)
- Max steps reached (default: 10)

## Usage

### Basic Usage

```python
from routing_board_game.routing_env import RoutingBoardGameEnv

# Create environment
env = RoutingBoardGameEnv(
    num_ai_pieces=8,      # Number of AI pieces
    max_steps=10,         # Max steps per episode
    reward_shaping=True,  # Enable reward shaping
    render_mode="human"   # Enable rendering
)

# Reset environment
obs, info = env.reset(seed=42)

# Take a step
action = 45  # Place a user piece at position (4, 5)
obs, reward, terminated, truncated, info = env.step(action)

# Render
env.render()
```

### Training an Agent

Train using the CLI:
```bash
# Train with default settings
python -m routing_board_game.train_routing

# Train with custom settings
python -m routing_board_game.train_routing \
    --num_ai_pieces 8 \
    --max_steps 10 \
    --total_timesteps 100000 \
    --algorithm PPO
```

Or using the nifty CLI (if installed):
```bash
nifty train-rl --num_ai_pieces 8 --max_steps 10 --total_timesteps 100000
```

### Observation Space

The observation is a dictionary containing:
- `board`: 10×10 grid with:
  - 0 = empty
  - 1 = AI piece (including user-placed)
  - 3 = root
- `step_count`: Current step number

### Action Space

Discrete action space with 100 actions (one for each board position):
- Action `n` places a user piece at position `(n // 10, n % 10)` which then routes toward the root
- Invalid placements (on occupied squares or root) are ignored

### AI Move Policies

AI movement is pluggable via a move policy:
- Default: `GreedyPolicy` (randomly selects among forward moves that reduce Manhattan distance)
- Custom: provide any `BaseMovePolicy` implementation to `RoutingBoardGameEnv(move_policy=...)` (e.g., wrap a neural net that scores forward moves).

## Testing

Run the comprehensive test suite:
```bash
python test_requirements.py
```

Run basic environment tests:
```bash
python test_routing_env.py
```

## Implementation Details

### Dynamic Blocking
The environment implements dynamic blocking semantics:
- A piece is blocked if it has no legal forward moves
- Blocking status is recomputed after each move
- Pieces can become unblocked as other pieces move out of the way

### AI Routing Phase
During the routing phase:
1. Iterate through pieces that haven't moved yet
2. Find pieces with available forward moves
3. Move one piece (randomly select if multiple moves available)
4. Update occupancy and recompute blocking status
5. Repeat until all pieces have moved or are blocked

## Files

- `routing_env.py` - Main Gymnasium environment implementation
- `train_routing.py` - Training script with PPO/DQN support
- `test_routing_env.py` - Basic functionality tests
- `test_requirements.py` - Comprehensive requirement validation tests

## Requirements

- Python 3.12+
- gymnasium
- numpy
- stable-baselines3 (for training)

## Architecture Notes

The environment is designed to be:
- **Headless**: No GUI dependencies
- **Stable**: Tested with random policies
- **Efficient**: Vectorizable for parallel training
- **Flexible**: Configurable number of pieces, steps, rewards
