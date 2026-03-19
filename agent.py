"""
agent.py — Double DQL agent with epsilon-greedy exploration.

Act:  ε-greedy — random with probability ε, best Q-value otherwise.
Learn: sample a batch from replay memory and train with Double DQN targets:
       Q_target = reward + γ × Q_target(next_state, argmax(Q_online(next_state)))
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf

from model import build_model, load_model, save_model, ACTION_SIZE, ONLINE_MODEL_PATH, TARGET_ONLINE_MODEL_PATH

# ── Settings ──────────────────────────────────────────────────────────────────

MEMORY_SIZE = 100_000     # replay buffer capacity
BATCH_SIZE = 128          # transitions per training step
GAMMA = 0.99              # discount factor
LEARN_EVERY = 4           # train once every N game steps
SYNC_TARGET_EVERY = 500   # copy online → target every N gradient updates
EPSILON_START = 1.0       # initial exploration rate
EPSILON_MIN = 0.01        # minimum exploration rate
EPSILON_DECAY = 0.99999   # per-step decay multiplier


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size ring buffer of (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Return a random batch as numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── DQL Agent ─────────────────────────────────────────────────────────────────

class DQLAgent:
    """Double DQN agent with experience replay and epsilon-greedy exploration."""

    def __init__(self):
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.grad_updates = 0
        self.step_count = 0
        self.huber = tf.keras.losses.Huber()

        # Resume from checkpoint if available
        saved = load_model(ONLINE_MODEL_PATH)
        if saved is not None:
            self.online_model = saved
            self.epsilon = EPSILON_MIN
        else:
            self.online_model = build_model()

        saved_target = load_model(TARGET_ONLINE_MODEL_PATH)
        self.target_model = saved_target if saved_target is not None else build_model()
        self.target_model.set_weights(self.online_model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """Pick an action: random (explore) or best Q-value (exploit)."""
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        q_values = self.online_model(state[np.newaxis], training=False).numpy()[0]
        return int(np.argmax(q_values))

    def step(self, state, action, reward, next_state, done) -> float:
        """
        Store transition, decay epsilon, and train every LEARN_EVERY steps.
        Returns loss (0.0 if no training happened).
        """
        self.memory.push(state, action, reward, next_state, done)
        self.step_count += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        if self.step_count % LEARN_EVERY != 0:
            return 0.0
        return self.learn()

    def learn(self) -> float:
        """
        Sample a batch and run one Double DQN update.

        Uses the online network to pick the best next action,
        and the target network to evaluate it. This reduces
        overestimation of Q-values.

        Returns loss, or 0.0 if not enough memory yet.
        """
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Double DQN: online picks action, target evaluates it
        best_actions = np.argmax(
            self.online_model(next_states, training=False).numpy(), axis=1
        )
        next_q = self.target_model(next_states, training=False).numpy()
        targets = rewards + GAMMA * next_q[np.arange(BATCH_SIZE), best_actions] * (1 - dones)

        # Build full target Q-value array (only overwrite the chosen action)
        target_q = self.online_model(states, training=False).numpy()
        target_q[np.arange(BATCH_SIZE), actions] = targets

        loss = self._train_step(tf.constant(states), tf.constant(target_q))

        self.grad_updates += 1
        if self.grad_updates % SYNC_TARGET_EVERY == 0:
            self.target_model.set_weights(self.online_model.get_weights())

        return float(loss)

    @tf.function
    def _train_step(self, states, target_q_values):
        """Single gradient update (compiled once by tf.function)."""
        with tf.GradientTape() as tape:
            predictions = self.online_model(states, training=True)
            loss = self.huber(target_q_values, predictions)
        grads = tape.gradient(loss, self.online_model.trainable_variables)
        self.online_model.optimizer.apply_gradients(
            zip(grads, self.online_model.trainable_variables)
        )
        return loss

    def save(self, path: str = ONLINE_MODEL_PATH) -> None:
        """Save both models to disk."""
        save_model(self.online_model, path)
        save_model(self.target_model, TARGET_ONLINE_MODEL_PATH)