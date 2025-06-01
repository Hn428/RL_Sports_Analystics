import pandas as pd
import matplotlib.pyplot as plt

# Reload the CSV file
file_path = "ppo_eval_results.csv"
results_df = pd.read_csv(file_path)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# === Plot Mean Episode Reward (Left Y-axis) ===
ax1.set_xlabel("Training Timesteps")
ax1.set_ylabel("Mean Episode Reward", color='black')
ax1.plot(results_df["checkpoint"], results_df["mean_reward"], color='black', label="Mean Reward", linewidth=2, marker='o')
for x, y in zip(results_df["checkpoint"], results_df["mean_reward"]):
    ax1.text(x, y, f"{y:.2f}", color='black', fontsize=8, ha='center', va='bottom')
ax1.tick_params(axis='y', labelcolor='black')

# === Plot AAE Metrics (Right Y-axis) ===
ax2 = ax1.twinx()
ax2.set_ylabel("Average Absolute Error", color='black')

# AAE lines - solid with markers
ax2.plot(results_df["checkpoint"], results_df["spread_aae"], color='red', linestyle='-', label="Spread AAE", linewidth=2, marker='o')
ax2.plot(results_df["checkpoint"], results_df["total_aae"], color='blue', linestyle='-', label="Total Points AAE", linewidth=2, marker='o')
for x, y in zip(results_df["checkpoint"], results_df["spread_aae"]):
    ax2.text(x, y, f"{y:.2f}", color='black', fontsize=8, ha='center', va='bottom')
for x, y in zip(results_df["checkpoint"], results_df["total_aae"]):
    ax2.text(x, y, f"{y:.2f}", color='black', fontsize=8, ha='center', va='bottom')

# Baselines - dotted red and blue
ax2.plot(results_df["checkpoint"], results_df["baseline_spread_aae"], color='red', linestyle=':', label="Baseline Spread AAE", linewidth=2)
ax2.plot(results_df["checkpoint"], results_df["baseline_total_aae"], color='blue', linestyle=':', label="Baseline Total AAE", linewidth=2)

ax2.tick_params(axis='y', labelcolor='black')

# === Combined Legend ===
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# === Title and Final Touches ===
plt.title("PPO Learning Progress Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
