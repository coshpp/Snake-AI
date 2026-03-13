"""
train.py — Runs the training loop.

Each episode:
  1. Reset the game
  2. Loop until the snake dies or starves:
       - Get the current state
       - Agent picks an action (random or best known)
       - Step the game — get next state, reward, done
       - Store the transition in memory
  3. Agent learns from a random sample of past transitions
  4. Log stats to CSV and print to console

Quit:
  Q or ESC — saves the model and exits cleanly
"""

import csv
import os
import sys
import numpy as np
import time

from snake_env import SnakeGame, FPS_AI, DEFAULT_GRID_SIZE
from agent import DQLAgent, BATCH_SIZE, GAMMA, LEARN_EVERY, EPSILON_DECAY
from model import HIDDEN_UNITS, ONLINE_MODEL_PATH, STATE_SIZE

# ── Settings ──────────────────────────────────────────────────────────────────

MAX_EPISODES = 500
SAVE_EVERY = 50
LOG_PATH = "training_log.csv"
RENDER = False  # set to True to watch training

if RENDER:
    import pygame
    from game import Renderer

# ── Training loop ─────────────────────────────────────────────────────────────

def train(grid_size):
    game = SnakeGame(grid_size=grid_size, human=False, speed=FPS_AI)
    renderer = Renderer(game) if RENDER else None

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

    log_file = open(LOG_PATH, "a", newline="")
    logger = csv.writer(log_file)
    if is_new_file:
        logger.writerow(["episode", "total_steps", "score", "record", "steps", "epsilon", "loss", "memory"])

    agent = DQLAgent()

    scores = []

    print(f"\n{'='*64}")
    print(f"  Deep Q-Learning Snake")
    print(f"  Grid : {grid_size}×{grid_size}   Starting episode : {start_episode}")
    print(f"  Episodes : {MAX_EPISODES}   Batch : {BATCH_SIZE}")
    print(f"  Memory : 100,000   γ : {GAMMA}")
    print(f"  Network : {STATE_SIZE} → {HIDDEN_UNITS} → {HIDDEN_UNITS} → 3")
    print(f"  Learn every : {LEARN_EVERY} steps   ε decay : {EPSILON_DECAY}/step")
    print(f"  Rendering : {'on' if RENDER else 'off'}")
    print(f"{'='*64}\n")
    if not RENDER:
        print("Training...".center(64))

    try:
        for episode in range(start_episode, MAX_EPISODES + 1):
            state = game.reset()
            done = False
            episode_loss = 0.0
            episode_trains = 0
            ep_start = time.time()

            # ── Play one full episode ──────────────────────────────────────────
            while not done:
                if RENDER:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            _shutdown(agent, log_file)
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
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

            # ── Episode over — log ────────────────────────────────────────────
            ep_time = time.time() - ep_start
            total_steps += game.steps
            scores.append(game.score)
            avg_loss = episode_loss / max(1, episode_trains)

            if episode % SAVE_EVERY == 0:
                agent.save()
                avg = np.mean(scores[-SAVE_EVERY:]) if len(scores) >= SAVE_EVERY else np.mean(scores)
                print(f"  ✓ Checkpoint saved (episode {episode}) — avg last {SAVE_EVERY}: {avg:.1f}")

            logger.writerow([
                episode,
                total_steps,
                game.score,
                game.record,
                game.steps,
                f"{agent.epsilon:.4f}",
                f"{avg_loss:.6f}",
                len(agent.memory),
            ])
            log_file.flush()

            print(
                f"EP: {episode:>3} | "
                f"Total Steps: {total_steps:>3} | "
                f"Score: {game.score:>3} | "
                f"Record: {game.record:>3} | "
                f"Steps: {game.steps:>4} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {ep_time:.4f}"
            )

    except KeyboardInterrupt:
        _shutdown(agent, log_file)

    agent.save()
    log_file.close()
    print(f"\nTraining complete — model saved to {ONLINE_MODEL_PATH}")
    if RENDER:
        pygame.quit()


def _shutdown(agent: DQLAgent, log_file):
    print("\nInterrupted — saving model...")
    agent.save()
    log_file.close()
    if RENDER:
        pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
    train(grid_size=grid_size)