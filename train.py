"""
train.py — Runs the training loop.

Each episode:
  1. Reset the game
  2. Loop until the snake dies or starves:
       - Get the current state (11 numbers)
       - Agent picks an action (random or best known)
       - Step the game — get next state, reward, done
       - Store the transition in memory
  3. Agent learns from a random sample of past transitions
  4. Epsilon decays — agent explores less next episode
  5. Log stats to CSV and print to console

Quit:
  Q or ESC — saves the model and exits cleanly
"""


import csv
import os
import sys
import numpy as np
import pygame

from snake_env import SnakeGame, FPS_AI, DEFAULT_GRID_SIZE
from game import Renderer
from agent import DQLAgent, BATCH_SIZE, GAMMA
from model import HIDDEN_UNITS, MODEL_PATH

# ── Settings ──────────────────────────────────────────────────────────────────

GRID_SIZE = DEFAULT_GRID_SIZE  
MAX_EPISODES = 500
TRAIN_EVERY = 1
SAVE_EVERY = 50
LOG_PATH = "training_log.csv"


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    game = SnakeGame(grid_size=GRID_SIZE, human=False, speed=FPS_AI)
    renderer = Renderer(game)
    agent = DQLAgent()

    is_new_file = not os.path.exists(LOG_PATH)
    start_episode = 1
    if not is_new_file:
        with open(LOG_PATH, "r") as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                start_episode = int(rows[-1][0]) + 1

    log_file = open(LOG_PATH, "a", newline="")
    logger = csv.writer(log_file)
    if is_new_file:
        logger.writerow(["episode", "score", "record", "steps", "epsilon", "loss", "memory"])

    scores = []
    last_loss = 0.0

    print(f"\n{'='*52}")
    print(f"  Deep Q-Learning Snake")
    print(f"  Grid : {GRID_SIZE}×{GRID_SIZE}   Starting episode : {start_episode}")
    print(f"  Episodes : {MAX_EPISODES}   Batch : {BATCH_SIZE}")
    print(f"  Memory : 100,000   γ : {GAMMA}")
    print(f"  Network : 11 → {HIDDEN_UNITS} → {HIDDEN_UNITS} → 3")
    print(f"{'='*52}\n")

    for episode in range(start_episode, start_episode + MAX_EPISODES):
        state = game.reset()
        done = False

        # ── Play one full episode ──────────────────────────────────────────
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _shutdown(agent, log_file)
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        _shutdown(agent, log_file)

            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            renderer.draw()

        # ── Episode over — now learn and log ──────────────────────────────
        scores.append(game.score)

        if episode % TRAIN_EVERY == 0:
            last_loss = agent.learn()

        agent.decay_epsilon()

        if episode % SAVE_EVERY == 0:
            agent.save()
            avg = np.mean(scores[-SAVE_EVERY:]) if len(scores) >= SAVE_EVERY else np.mean(scores)
            print(f"  ✓ Checkpoint saved (episode {episode}) — avg last {SAVE_EVERY}: {avg:.1f}")

        logger.writerow([
            episode, game.score, game.record, game.steps,
            f"{agent.epsilon:.4f}", f"{last_loss:.6f}", len(agent.memory),
        ])
        log_file.flush()

        print(
            f"EP {episode:>4} | "
            f"Score {game.score:>3} | "
            f"Record {game.record:>3} | "
            f"ε {agent.epsilon:.3f} | "
            f"Loss {last_loss:.4f}"
        )

    agent.save()
    log_file.close()
    print(f"\nTraining complete — model saved to {MODEL_PATH}")
    pygame.quit()


def _shutdown(agent: DQLAgent, log_file):
    print("\nInterrupted — saving model...")
    agent.save()
    log_file.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    train()