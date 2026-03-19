"""
train.py — Training loop.

Each episode:
  1. Reset the game
  2. Play until the snake dies or starves
  3. Agent learns from a random sample of past transitions
  4. Log stats to CSV and print to console

Quit: Ctrl+C — saves the model and exits cleanly.
"""

import csv
import os
import sys
import time

import numpy as np

from snake_env import SnakeGame, FPS_AI, DEFAULT_GRID_SIZE
from agent import DQLAgent, BATCH_SIZE, GAMMA, LEARN_EVERY, EPSILON_DECAY
from model import ONLINE_MODEL_PATH, STATE_SIZE

# ── Settings ──────────────────────────────────────────────────────────────────

MAX_EPISODES = 15_000
SAVE_EVERY = 50
LOG_PATH = "training_log.csv"
RENDER = False

if RENDER:
    import pygame
    from game import Renderer

# ── Training loop ─────────────────────────────────────────────────────────────

def train(grid_size):
    game = SnakeGame(grid_size=grid_size, human=False, speed=FPS_AI)
    renderer = Renderer(game) if RENDER else None

    # Resume from existing log if present
    is_new_file = not os.path.exists(LOG_PATH)
    start_episode = 1
    record = 0
    total_steps = 0

    if not is_new_file:
        with open(LOG_PATH, "r", newline="") as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                header = rows[0]
                last_row = rows[-1]
                start_episode = int(last_row[0]) + 1
                record = max(int(row[2]) for row in rows[1:])
                if "total_steps" in header:
                    total_steps = int(last_row[header.index("total_steps")])
                else:
                    total_steps = sum(int(row[3]) for row in rows[1:])

    game.record = record
    agent = DQLAgent()
    scores = []

    log_file = open(LOG_PATH, "a", newline="")
    logger = csv.writer(log_file)
    if is_new_file:
        logger.writerow([
            "episode", "total_steps", "score", "record",
            "steps", "epsilon", "loss", "memory",
        ])

    # Print training config
    print(f"\n{'='*64}")
    print(f"  Deep Q-Learning Snake")
    print(f"  Grid: {grid_size}×{grid_size}   Starting episode: {start_episode}")
    print(f"  Episodes: {MAX_EPISODES}   Batch: {BATCH_SIZE}   γ: {GAMMA}")
    print(f"  State: {STATE_SIZE}   Learn every: {LEARN_EVERY} steps")
    print(f"  ε decay: {EPSILON_DECAY}/step   Render: {'on' if RENDER else 'off'}")
    print(f"{'='*64}\n")

    try:
        for episode in range(start_episode, MAX_EPISODES + 1):
            state = game.reset()
            done = False
            episode_loss = 0.0
            episode_trains = 0
            ep_start = time.time()

            while not done:
                if RENDER:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            _shutdown(agent, log_file)

                action = agent.act(state)
                next_state, reward, done = game.step(action)
                loss = agent.step(state, action, reward, next_state, done)

                if loss > 0:
                    episode_loss += loss
                    episode_trains += 1

                state = next_state

                if RENDER:
                    renderer.draw()

            # ── Episode over — log ────────────────────────────────────────
            ep_time = time.time() - ep_start
            total_steps += game.steps
            scores.append(game.score)
            avg_loss = episode_loss / max(1, episode_trains)

            if episode % SAVE_EVERY == 0:
                agent.save()
                window = scores[-SAVE_EVERY:] if len(scores) >= SAVE_EVERY else scores
                print(f"  ✓ Checkpoint (ep {episode}) — avg last {len(window)}: {np.mean(window):.1f}")

            logger.writerow([
                episode, total_steps, game.score, game.record,
                game.steps, f"{agent.epsilon:.4f}",
                f"{avg_loss:.6f}", len(agent.memory),
            ])
            log_file.flush()

            print(
                f"EP: {episode:>5} | "
                f"Steps: {total_steps:>8} | "
                f"Score: {game.score:>3} | "
                f"Record: {game.record:>3} | "
                f"Ep Steps: {game.steps:>4} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {ep_time:.2f}s"
            )

    except KeyboardInterrupt:
        _shutdown(agent, log_file)

    agent.save()
    log_file.close()
    print(f"\nTraining complete — model saved to {ONLINE_MODEL_PATH}")
    if RENDER:
        pygame.quit()


def _shutdown(agent: DQLAgent, log_file):
    """Save and exit cleanly on interrupt."""
    print("\nInterrupted — saving model...")
    agent.save()
    log_file.close()
    if RENDER:
        pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
    train(grid_size=grid_size)