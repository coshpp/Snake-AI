# Snake-AI

This project implements a Deep Q-Learning (DQL) agent that learns to play Snake using reinforcement learning. The model represents the environment using an 11-dimensional state space, which encodes immediate dangers (straight, left, right), current movement direction, and the relative position of the food. 

Because the state representation is relative rather than tied to absolute board coordinates, the model is scalable to larger board sizes without architectural changes. However, this simplified 11-state encoding also limits performance: the agent only has local awareness and cannot reason about the full board layout, long-term path planning, or complex trapping scenarios. As a result, performance eventually plateaus despite continued training.
