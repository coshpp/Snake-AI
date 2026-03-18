# Snake AI — Deep Q-Learning

A Deep Q-Learning (DQL) agent that learns to play Snake through reinforcement learning. The agent starts with zero knowledge and learns to navigate, find food, and avoid trapping itself through trial and error.

## How It Works

The agent uses a **Double Deep Q-Network (DQN) ** architecture: two neural networks (online and target) that work together to learn Q-values. Q-values are estimates of how good each action is in a given state. The online network picks actions and trains, while the target network provides stable training targets and syncs periodically.

**Epsilon-greedy exploration** balances learning: the agent starts fully random (ε = 1.0) and gradually shifts to exploiting its best known strategy (ε → 0.01) as it gains experience.

### State Representation (24 features)

The agent sees the game through a 24-dimensional state vector, all relative to its current heading:

| Features | Count | Description |
|----------|-------|-------------|
| **Raycasts** | 16 | 8 rays (forward, diagonals, sides, backward) × 2 signals each (wall distance, body distance). Each signal is `1/distance`, giving stronger response to closer obstacles. |
| **Food direction** | 2 | Where the food is relative to the snake's heading (forward distance, right distance). |
| **Reachable space** | 3 | For each possible action (straight, right, left): what fraction of free cells are reachable via BFS. Tells the agent which moves lead to open space vs dead ends. |
| **Tail reachable** | 3 | For each possible action: can the snake still reach its own tail? |

Because the state is relative to the snake's heading (not absolute grid coordinates), the agent generalizes across directions and scales to different grid sizes without retraining.

### Reward Structure

| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Die (collision or starvation) | -10 |
| Normal step | -0.01 |
| Move toward food (fading) | +0.1 × weight (fades to 0 by length 50) |
| Move away from food (fading) | -0.1 × weight (fades to 0 by length 50) |

The proximity shaping fades linearly from full strength at length 3 to zero at length 50. This teaches early food-seeking without misleading the agent when the snake is long and direct paths run through its own body.

### Network Architecture

A simple feedforward network:

```
24 inputs → 256 → 256 → 3 outputs (Q-values for straight, right, left)
```

Trained with Huber loss and Adam optimizer. A `@tf.function`-compiled training step avoids TensorFlow graph retracing overhead.

### Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| γ (discount) | 0.99 | High discount so the agent values long-term survival. A death 40 steps away still has significant impact. |
| Batch size | 128 | Large enough for stable gradients, small enough for fast updates. |
| Replay buffer | 100,000 | Stores diverse past transitions so the agent doesn't just learn from recent games. |
| Learn every | 4 steps | Collects a few transitions between updates so training batches are more varied. |
| Target sync | Every 500 updates | Keeps the target network stable to prevent Q-value oscillation. Frequent enough to track improvement. |
| ε decay | 0.99999/step | Slow decay gives ~460k steps of exploration before settling into exploitation. |
| BFS cap | 150 cells | Limits flood fill cost at high scores. Tail search continues past the cap for accuracy. |

## Project Structure

```
agent.py      — Double DQN agent with replay buffer
model.py      — Neural network definition (build, save, load)
snake_env.py  — Game logic, state construction, reward shaping
train.py      — Training loop with CSV logging and checkpoints
play_ai.py    — Watch the trained agent play (no training)
game.py       — Pygame renderer and human play mode
visualizer.py — Training data visualization
```

## Usage

**Train:**
```bash
python3 train.py              # default 20×20 grid
python3 train.py <size>       # size×size grid
```

Training saves checkpoints every 50 episodes. Ctrl+C saves and exits cleanly. Progress is logged to `training_log.csv`.

**Watch the AI play:**
```bash
python3 play_ai.py            # default 20x20 grid
python3 play_ai.py <size>     # size×size grid
```

Use ↑/↓ to change speed, Space to pause and inspect Q-values, R to reset.

**Play as a human:**
```bash
python3 game.py
```

WASD or arrow keys to move.

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pygame
- Pandas
- Matplotlib
