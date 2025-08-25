import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'rl_learning_curve_20000.csv'
df = pd.read_csv(file_path)

# Calculate the rolling 200 average of the rewards column
df['rolling_reward'] = df['reward'].rolling(window=200).mean()

# Drop NaN values from rolling average for trend line fitting
valid_indices = df['rolling_reward'].dropna().index
x = valid_indices.to_numpy()
y = df.loc[valid_indices, 'rolling_reward'].to_numpy()

# Fit a linear trend line
coefficients = np.polyfit(x, y, deg=1)
trend_line = np.poly1d(coefficients)

# Plot the rolling average with vertical lines and trend line
plt.figure(figsize=(14, 6))
plt.plot(df['rolling_reward'], label='Rolling 200 Average of Rewards')
plt.plot(x, trend_line(x), color='orange', linestyle='-', linewidth=2, label='Trend Line')
plt.axvline(x=3000, color='red', linestyle='--', label='Epsilon = 1 (Episode 3000)')
plt.axvline(x=10000, color='green', linestyle='--', label='Epsilon = 0.01 (Episode 10000)')
plt.title('Rolling 200 Average of Rewards with Trend Line and Epsilon Milestones')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()