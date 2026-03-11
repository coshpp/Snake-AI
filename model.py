"""
model.py — The neural network 

Architecture:
  11 inputs → 256 neurons → 256 neurons → 3 outputs
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ── Settings ──────────────────────────────────────────────────────────────────

STATE_SIZE = 11     
ACTION_SIZE = 3     
HIDDEN_UNITS = 256  
LEARNING_RATE = 0.001  
ONLINE_MODEL_PATH = "snake_online_model.keras"
TARGET_ONLINE_MODEL_PATH = "snake_target_model.keras"


def build_model() -> Sequential:
    model = Sequential([
        Dense(HIDDEN_UNITS, activation="relu", input_shape=(STATE_SIZE,)),
        Dense(HIDDEN_UNITS, activation="relu"),
        Dense(ACTION_SIZE, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="Huber")
    return model


def load_model(path: str = ONLINE_MODEL_PATH) -> Sequential:
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        print(f"Loaded model from {path}")
        return model
    except Exception as e:
        print(f"Could not load model: {e}")
        return None


def save_model(model: Sequential, path: str = ONLINE_MODEL_PATH):
    model.save(path)
