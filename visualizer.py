import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

# change to 4 for steps
steps_data = df.iloc[:, 1] 

# Alternatively, since you have headers, you can just call it by name:
# steps_data = df["steps"]

# 3. Plot it
plt.plot(steps_data, color="blue", alpha=0.6)
plt.title("Snake Survival (Score per Episode)")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.grid(True)
plt.show()