import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Loss vs Episodes
ax1.plot(df["loss"], color="red", alpha=0.3, label="Loss")
ax1.plot(df["loss"].rolling(50).mean(), color="red", label="50-ep avg")
ax1.set_title("Loss per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Average Score vs Episodes
ax2.plot(df["score"], color="blue", alpha=0.3, label="Score")
ax2.plot(df["score"].rolling(50).mean(), color="blue", label="50-ep avg")
ax2.set_title("Score per Episode")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Score")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()