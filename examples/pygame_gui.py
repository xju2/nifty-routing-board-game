#!/usr/bin/env python3
"""
Pygame-based GUI to visualize the routing board game.

Controls:
  - Left click: place a user piece on the clicked cell
  - Space: place a random user piece
  - R: reset the environment
  - Q or ESC: quit

After each action the AI moves its pieces sequentially. The GUI highlights each
piece as it moves (source red, destination green) so you can see the per-piece
updates within a turn.
"""

import sys
import pygame

from routing_board_game.routing_env import RoutingBoardGameEnv


CELL_SIZE = 60
MARGIN = 2
SIDEBAR_WIDTH = 240
FPS = 60
MOVE_DELAY_MS = 420

# Colors
BG_COLOR = (20, 24, 28)
GRID_COLOR = (60, 70, 80)
AI_COLOR = (66, 179, 245)
ROOT_COLOR = (255, 204, 0)
TEXT_COLOR = (225, 230, 235)
HIGHLIGHT_SRC = (232, 80, 91)
HIGHLIGHT_DST = (80, 200, 120)


def draw_board(screen, font, board, root_pos, step_info, highlight=None):
    """Draw the grid, pieces, and sidebar info."""
    screen.fill(BG_COLOR)
    grid_size = board.shape[0]

    # Draw cells
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * CELL_SIZE + MARGIN
            y = row * CELL_SIZE + MARGIN
            rect = pygame.Rect(x, y, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

            pos = (row, col)
            # Root
            if pos == root_pos:
                pygame.draw.rect(screen, ROOT_COLOR, rect)
            # AI piece
            if board[row, col] == 1:
                pygame.draw.ellipse(screen, AI_COLOR, rect.inflate(-10, -10))

    # Highlight current move
    if highlight:
        src, dst = highlight
        for pos, color in [(src, HIGHLIGHT_SRC), (dst, HIGHLIGHT_DST)]:
            x = pos[1] * CELL_SIZE + MARGIN
            y = pos[0] * CELL_SIZE + MARGIN
            rect = pygame.Rect(x, y, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)
            pygame.draw.rect(screen, color, rect, 3)

        # Draw a line from src to dst
        src_center = (
            src[1] * CELL_SIZE + CELL_SIZE // 2,
            src[0] * CELL_SIZE + CELL_SIZE // 2,
        )
        dst_center = (
            dst[1] * CELL_SIZE + CELL_SIZE // 2,
            dst[0] * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.line(screen, HIGHLIGHT_SRC, src_center, dst_center, 4)

    # Sidebar
    sidebar_x = grid_size * CELL_SIZE + 20
    lines = [
        f"Step: {step_info.get('step', 0)} / {step_info.get('max_steps', '?')}",
        f"Reward: {step_info.get('reward', 0):.2f}",
        f"Pieces remaining: {step_info.get('pieces_remaining', 0)}",
        f"Pieces routed: {step_info.get('pieces_routed_total', 0)}",
        f"Pieces spawned: {step_info.get('pieces_spawned_total', 0)}",
        f"User pieces added: {step_info.get('user_pieces_added', 0)}",
        "",
        f"Placement success: {step_info.get('placement_success', False)}",
        f"Terminated: {step_info.get('terminated', False)}",
        f"Truncated: {step_info.get('truncated', False)}",
        "",
        "Controls:",
        " Click: place piece",
        " Space: random action",
        " R: reset",
        " Q / Esc: quit",
    ]

    for i, line in enumerate(lines):
        text = font.render(line, True, TEXT_COLOR)
        screen.blit(text, (sidebar_x, 20 + i * 24))

    pygame.display.flip()


def animate_moves(
    screen,
    font,
    board_before,
    root_pos,
    step_info,
    move_sequence,
    placement_pos=None,
):
    """Animate the sequence of AI moves for the current turn."""
    temp_board = board_before.copy()
    if placement_pos is not None and temp_board[placement_pos] == 0:
        # Show the newly placed piece before any AI moves occur
        temp_board[placement_pos] = 1

    draw_board(screen, font, temp_board, root_pos, step_info)
    pygame.time.delay(MOVE_DELAY_MS // 2)

    for src, dst in move_sequence:
        # Show highlight before move
        draw_board(screen, font, temp_board, root_pos, step_info, highlight=(src, dst))
        pygame.time.delay(MOVE_DELAY_MS)

        # Apply move on the temp board for visualization
        if temp_board[src] == 1:
            temp_board[src] = 0
        if dst != root_pos:
            temp_board[dst] = 1

        draw_board(screen, font, temp_board, root_pos, step_info, highlight=(src, dst))
        pygame.time.delay(MOVE_DELAY_MS)


def run_gui():
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)

    env = RoutingBoardGameEnv(render_mode=None)
    obs, info = env.reset(seed=42)

    board = obs["board"]
    grid_size = env.grid_size

    screen = pygame.display.set_mode(
        (grid_size * CELL_SIZE + SIDEBAR_WIDTH, grid_size * CELL_SIZE)
    )
    pygame.display.set_caption("Routing Board Game - Pygame Viewer")

    last_reward = 0.0
    last_info = info
    last_terminated = False
    last_truncated = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    board = obs["board"]
                    last_reward = 0.0
                    last_info = info
                    last_terminated = False
                    last_truncated = False
                elif event.key == pygame.K_SPACE and not (
                    last_terminated or last_truncated
                ):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    placement_pos = (
                        (action // env.grid_size, action % env.grid_size)
                        if info["placement_success"]
                        else None
                    )
                    step_info = {
                        "step": info["step_count"],
                        "max_steps": env.max_steps,
                        "reward": reward,
                        **info,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    animate_moves(
                        screen,
                        font,
                        board,
                        env.root_pos,
                        step_info,
                        info["move_sequence"],
                        placement_pos=placement_pos,
                    )
                    board = obs["board"]
                    last_reward = reward
                    last_info = info
                    last_terminated = terminated
                    last_truncated = truncated
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if last_terminated or last_truncated:
                    continue
                mouse_x, mouse_y = pygame.mouse.get_pos()
                col = mouse_x // CELL_SIZE
                row = mouse_y // CELL_SIZE

                if 0 <= row < grid_size and 0 <= col < grid_size:
                    action = row * grid_size + col
                    obs, reward, terminated, truncated, info = env.step(action)
                    placement_pos = (row, col) if info["placement_success"] else None
                    step_info = {
                        "step": info["step_count"],
                        "max_steps": env.max_steps,
                        "reward": reward,
                        **info,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    animate_moves(
                        screen,
                        font,
                        board,
                        env.root_pos,
                        step_info,
                        info["move_sequence"],
                        placement_pos=placement_pos,
                    )
                    board = obs["board"]
                    last_reward = reward
                    last_info = info
                    last_terminated = terminated
                    last_truncated = truncated

        # Update display
        current_info = {
            "step": last_info.get("step_count", 0),
            "max_steps": env.max_steps,
            "reward": last_reward,
            **last_info,
            "terminated": last_terminated,
            "truncated": last_truncated,
        }
        draw_board(screen, font, board, env.root_pos, current_info)
        clock.tick(FPS)

        if last_terminated or last_truncated:
            # Brief pause so the user can see the end state
            pygame.time.delay(400)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui()
