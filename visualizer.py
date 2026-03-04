import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV
df = pd.read_csv("training_log.csv")

# 2. Extract the 4th column 
# (iloc uses 0-based indexing, so 3 is the 4th column)
steps_data = df.iloc[:, 3] 

# Alternatively, since you have headers, you can just call it by name:
# steps_data = df["steps"]

# 3. Plot it
plt.plot(steps_data, color="blue", alpha=0.6)
plt.title("Snake Survival (Steps per Episode)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.show()