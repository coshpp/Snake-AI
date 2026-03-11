"""
agent.py — The DQL agent.

The agent does two things:

  1. Act — decide what to do given the current state.
     Uses epsilon-greedy: flip a coin weighted by epsilon.
       - heads (probability ε): pick a random action  → explore
       - tails (probability 1-ε): pick the best action → exploit
     Epsilon starts at 1.0 (always random) and decays toward 0.01
     (almost always picks the best known action).

  2. Learn — after each episode, look back at past experience and
     improve the network's Q-value predictions.
     Samples a random batch from memory and trains using:
       Q_target = reward + γ × max(Q(next_state))
"""

import random
from collections import deque

import numpy as np

from model import build_model, load_model, save_model, ACTION_SIZE, ONLINE_MODEL_PATH, TARGET_ONLINE_MODEL_PATH

# ── Settings ──────────────────────────────────────────────────────────────────

MEMORY_SIZE = 100_000  # how many past transitions to remember
BATCH_SIZE = 128      # how many to sample per learning step
GAMMA = 0.9            # discount factor — how much to value future rewards
EPSILON_START = 1.0    # start fully random
EPSILON_MIN = 0.01     # never go below 1% random
EPSILON_DECAY = 0.9991  # multiply epsilon by this after each episode


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores past game transitions so the agent can learn from them later.

    Each transition is one moment in the game:
      (state, action, reward, next_state, done)

    Sample randomly when learning to prevent overfitting
    """

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQL Agent ─────────────────────────────────────────────────────────────────

class DQLAgent:
    def __init__(self):
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START

        self.steps = 0

        # Resume from a saved model if one exists
        saved = load_model(ONLINE_MODEL_PATH)
        if saved is not None:
            self.online_model = saved
            self.epsilon = 0.1
        else:
            self.online_model = build_model()

        saved_target = load_model(TARGET_ONLINE_MODEL_PATH)
        self.target_model = saved_target if saved_target is not None else build_model()
        self.target_model.set_weights(self.online_model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """
        Pick an action for the given state.
        Random action if exploring, best known action if exploiting.
        """
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)  # explore
        q_values = self.online_model(state[np.newaxis], training=False).numpy()[0]
        return int(np.argmax(q_values))  # exploit — pick highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Save a transition to memory."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """
        Sample a batch from memory and update the network.

        For each transition in the batch:
          - Compute the Q-target: reward + γ × best future Q-value
            (for terminal states — death/starvation — there's no future,
             so the target is just the reward)
          - Replace the network's prediction for the chosen action with
            this target, leave the other two actions unchanged
          - Run one round of backprop to push predictions toward targets
        
        Returns the loss (how wrong the predictions were on average).
        Returns 0.0 if the buffer doesn't have enough transitions yet.
        """
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        best_actions = np.argmax(self.online_model(next_states, training=False).numpy(), axis=1)
        next_q = self.target_model(next_states, training=False).numpy()
        targets = rewards + GAMMA * next_q[np.arange(BATCH_SIZE), best_actions] * (1 - dones)
        predicted_q_values = self.online_model(states, training=False).numpy()
        target_q_values = predicted_q_values.copy()
        target_q_values[np.arange(BATCH_SIZE), actions] = targets

        history = self.online_model.fit(
            states, target_q_values,
            batch_size=BATCH_SIZE,
            epochs=1, verbose=0,
        )

        self.steps += 1
        if self.steps % 100 == 0:
            self.target_model.set_weights(self.online_model.get_weights())

        return float(history.history["loss"][0])

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path: str = ONLINE_MODEL_PATH):
        save_model(self.online_model, path)
        save_model(self.target_model, TARGET_ONLINE_MODEL_PATH)
