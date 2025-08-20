import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, gaussian_kde

# Load your data
df = pd.read_csv("agent_summary_statistics.csv")

# Ensure 'final_balance' is numeric and drop invalids
df['final_balance'] = pd.to_numeric(df['final_balance'], errors='coerce')
df = df[np.isfinite(df['final_balance'])]

# Set agent type to compare
target_agent = "RLBettingAgent"
agent_types = df['agent_type'].unique()

# Extract only valid agents (excluding target agent for pairing)
comparison_agents = [a for a in agent_types if a != target_agent]

# Begin analysis
for agent in comparison_agents:
    rl_data = df[df['agent_type'] == target_agent]['final_balance'].astype(np.float64).values
    comp_data = df[df['agent_type'] == agent]['final_balance'].astype(np.float64).values

    # KDE
    rl_kde = gaussian_kde(rl_data)
    comp_kde = gaussian_kde(comp_data)
    xmin = min(rl_data.min(), comp_data.min())
    xmax = max(rl_data.max(), comp_data.max())
    x_vals = np.linspace(xmin, xmax, 500)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"RL Agent vs {agent}", fontsize=16)

    # KDE plot
    axs[0].plot(x_vals, rl_kde(x_vals), label='RL Agent')
    axs[0].plot(x_vals, comp_kde(x_vals), label=agent)
    axs[0].fill_between(x_vals, rl_kde(x_vals), alpha=0.3)
    axs[0].fill_between(x_vals, comp_kde(x_vals), alpha=0.3)
    axs[0].set_title("KDE of Final Balances")
    axs[0].legend()

    # Boxplot
    sns.boxplot(data=[rl_data, comp_data], ax=axs[1])
    axs[1].set_xticklabels(['RL Agent', agent])
    axs[1].set_ylabel('Final Balance')
    axs[1].set_title('Boxplot of Final Balances')

    plt.tight_layout()
    plt.show()